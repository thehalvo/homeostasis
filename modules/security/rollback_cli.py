"""
CLI commands for rollback management.

Provides quick 'undo' or 'rollback' commands for both CLI and dashboard integration.
"""

import argparse
import sys
import json
from typing import Optional, List
from pathlib import Path

from .rollback_manager import (
    RollbackManager, RollbackTrigger, create_rollback_manager,
    SecurityMonitoringConfig
)


def cmd_create_snapshot(args: argparse.Namespace) -> None:
    """Handle the create-snapshot command."""
    rollback_manager = create_rollback_manager()
    
    try:
        file_paths = args.files if args.files else []
        project_root = Path(args.project_root) if args.project_root else None
        
        snapshot_id = rollback_manager.create_snapshot(
            context_id=args.context_id,
            file_paths=file_paths,
            project_root=project_root
        )
        
        print(f"✓ Created snapshot: {snapshot_id}")
        
        if args.json:
            output = {
                "snapshot_id": snapshot_id,
                "context_id": args.context_id,
                "files_count": len(file_paths)
            }
            print(json.dumps(output, indent=2))
            
    except Exception as e:
        print(f"Error creating snapshot: {e}")
        sys.exit(1)


def cmd_rollback(args: argparse.Namespace) -> None:
    """Handle the rollback command."""
    rollback_manager = create_rollback_manager()
    
    try:
        # Determine trigger type
        if args.trigger:
            trigger = RollbackTrigger(args.trigger)
        else:
            trigger = RollbackTrigger.MANUAL
        
        rollback_id = rollback_manager.rollback_patch(
            snapshot_id=args.snapshot_id,
            trigger=trigger,
            triggered_by=args.user or "cli_user",
            reason=args.reason
        )
        
        print(f"✓ Rollback completed: {rollback_id}")
        
        if args.json:
            rollback_op = rollback_manager.rollback_operations[rollback_id]
            output = {
                "rollback_id": rollback_id,
                "snapshot_id": args.snapshot_id,
                "status": rollback_op.status.value,
                "files_restored": len(rollback_op.files_restored),
                "completed_at": rollback_op.completed_at
            }
            print(json.dumps(output, indent=2))
            
    except Exception as e:
        print(f"Error during rollback: {e}")
        sys.exit(1)


def cmd_undo(args: argparse.Namespace) -> None:
    """Handle the undo command (quick rollback by context ID)."""
    rollback_manager = create_rollback_manager()
    
    try:
        # Find the most recent snapshot for the context
        context_snapshots = [
            snapshot for snapshot in rollback_manager.snapshots.values()
            if snapshot.context_id == args.context_id
        ]
        
        if not context_snapshots:
            print(f"No snapshots found for context: {args.context_id}")
            sys.exit(1)
        
        # Get the most recent snapshot
        latest_snapshot = max(context_snapshots, key=lambda s: s.timestamp)
        
        # Confirm if not forcing
        if not args.force:
            print(f"About to rollback using snapshot: {latest_snapshot.snapshot_id}")
            print(f"Files to restore: {len(latest_snapshot.file_paths)}")
            print(f"Snapshot created: {latest_snapshot.timestamp}")
            
            response = input("Continue with rollback? (y/N): ").lower().strip()
            if response not in ['y', 'yes']:
                print("Rollback cancelled")
                return
        
        rollback_id = rollback_manager.rollback_patch(
            snapshot_id=latest_snapshot.snapshot_id,
            trigger=RollbackTrigger.MANUAL,
            triggered_by=args.user or "cli_user",
            reason=f"Quick undo for context {args.context_id}"
        )
        
        print(f"✓ Undo completed: {rollback_id}")
        print(f"Restored {len(rollback_manager.rollback_operations[rollback_id].files_restored)} files")
        
    except Exception as e:
        print(f"Error during undo: {e}")
        sys.exit(1)


def cmd_list_snapshots(args: argparse.Namespace) -> None:
    """Handle the list-snapshots command."""
    rollback_manager = create_rollback_manager()
    
    try:
        snapshots = list(rollback_manager.snapshots.values())
        
        # Filter by context ID if specified
        if args.context_id:
            snapshots = [s for s in snapshots if s.context_id == args.context_id]
        
        # Sort by timestamp (newest first)
        snapshots.sort(key=lambda s: s.timestamp, reverse=True)
        
        if not snapshots:
            print("No snapshots found")
            return
        
        if args.json:
            output = [
                {
                    "snapshot_id": s.snapshot_id,
                    "context_id": s.context_id,
                    "timestamp": s.timestamp,
                    "files_count": len(s.file_paths),
                    "backup_location": s.backup_location,
                    "git_commit": s.git_commit_hash
                }
                for s in snapshots[:args.limit]
            ]
            print(json.dumps(output, indent=2))
        else:
            print("Snapshots:")
            print("=" * 80)
            
            for snapshot in snapshots[:args.limit]:
                print(f"ID: {snapshot.snapshot_id}")
                print(f"Context: {snapshot.context_id}")
                print(f"Created: {snapshot.timestamp}")
                print(f"Files: {len(snapshot.file_paths)}")
                if snapshot.git_commit_hash:
                    print(f"Git Commit: {snapshot.git_commit_hash[:8]}...")
                print("-" * 40)
                
    except Exception as e:
        print(f"Error listing snapshots: {e}")
        sys.exit(1)


def cmd_list_rollbacks(args: argparse.Namespace) -> None:
    """Handle the list-rollbacks command."""
    rollback_manager = create_rollback_manager()
    
    try:
        rollbacks = rollback_manager.get_rollback_history(args.context_id)
        
        if not rollbacks:
            print("No rollback operations found")
            return
        
        # Limit results
        rollbacks = rollbacks[:args.limit]
        
        if args.json:
            output = [
                {
                    "rollback_id": r.rollback_id,
                    "snapshot_id": r.snapshot_id,
                    "context_id": r.context_id,
                    "trigger": r.trigger.value,
                    "status": r.status.value,
                    "triggered_at": r.triggered_at,
                    "triggered_by": r.triggered_by,
                    "files_restored": len(r.files_restored)
                }
                for r in rollbacks
            ]
            print(json.dumps(output, indent=2))
        else:
            print("Rollback Operations:")
            print("=" * 80)
            
            for rollback in rollbacks:
                print(f"ID: {rollback.rollback_id}")
                print(f"Context: {rollback.context_id}")
                print(f"Trigger: {rollback.trigger.value}")
                print(f"Status: {rollback.status.value}")
                print(f"Triggered: {rollback.triggered_at}")
                print(f"By: {rollback.triggered_by}")
                print(f"Files Restored: {len(rollback.files_restored)}")
                if rollback.error_message:
                    print(f"Error: {rollback.error_message}")
                print("-" * 40)
                
    except Exception as e:
        print(f"Error listing rollbacks: {e}")
        sys.exit(1)


def cmd_rollback_status(args: argparse.Namespace) -> None:
    """Handle the rollback-status command."""
    rollback_manager = create_rollback_manager()
    
    try:
        summary = rollback_manager.get_rollback_summary()
        
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print("Rollback System Status:")
            print("=" * 50)
            print(f"Total Snapshots: {summary['total_snapshots']}")
            print(f"Total Rollbacks: {summary['total_rollbacks']}")
            print(f"Success Rate: {summary['success_rate']:.1%}")
            print(f"Storage Usage: {summary['storage_usage_mb']:.1f} MB")
            print()
            
            if summary['rollbacks_by_trigger']:
                print("Rollbacks by Trigger:")
                for trigger, count in summary['rollbacks_by_trigger'].items():
                    print(f"  {trigger}: {count}")
                print()
            
            if summary['recent_rollbacks']:
                print("Recent Rollbacks:")
                for rollback in summary['recent_rollbacks'][:5]:
                    print(f"  {rollback['rollback_id'][:8]}... - {rollback['trigger']} - {rollback['status']}")
                
    except Exception as e:
        print(f"Error getting rollback status: {e}")
        sys.exit(1)


def cmd_cleanup_snapshots(args: argparse.Namespace) -> None:
    """Handle the cleanup-snapshots command."""
    rollback_manager = create_rollback_manager()
    
    try:
        if not args.force:
            print(f"About to clean up snapshots older than {args.retention_days} days")
            response = input("Continue? (y/N): ").lower().strip()
            if response not in ['y', 'yes']:
                print("Cleanup cancelled")
                return
        
        cleaned_count = rollback_manager.cleanup_old_snapshots(args.retention_days)
        print(f"✓ Cleaned up {cleaned_count} old snapshots")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        sys.exit(1)


def cmd_monitor(args: argparse.Namespace) -> None:
    """Handle the monitor command for automatic rollback monitoring."""
    rollback_manager = create_rollback_manager()
    
    try:
        print(f"Starting monitoring for context {args.context_id}...")
        print(f"Will monitor for {args.duration} minutes")
        print("Press Ctrl+C to stop monitoring")
        
        rollback_id = rollback_manager.monitor_and_auto_rollback(
            context_id=args.context_id,
            snapshot_id=args.snapshot_id,
            monitoring_duration_minutes=args.duration
        )
        
        if rollback_id:
            print(f"⚠️  Auto-rollback triggered: {rollback_id}")
            rollback_op = rollback_manager.rollback_operations[rollback_id]
            print(f"Trigger: {rollback_op.trigger.value}")
            print(f"Reason: {rollback_op.metadata.get('reason', 'Unknown')}")
        else:
            print("✓ Monitoring completed - no issues detected")
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Error during monitoring: {e}")
        sys.exit(1)


def create_rollback_cli_parser() -> argparse.ArgumentParser:
    """
    Create the CLI parser for rollback commands.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Homeostasis Rollback Management CLI",
        prog="homeostasis rollback"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # create-snapshot command
    snapshot_parser = subparsers.add_parser(
        "create-snapshot",
        help="Create a snapshot before applying patches"
    )
    snapshot_parser.add_argument(
        "context_id",
        help="Context ID for the patch"
    )
    snapshot_parser.add_argument(
        "--files", "-f",
        nargs="+",
        help="List of files to include in snapshot"
    )
    snapshot_parser.add_argument(
        "--project-root", "-p",
        help="Root directory of the project"
    )
    snapshot_parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    snapshot_parser.set_defaults(func=cmd_create_snapshot)
    
    # rollback command
    rollback_parser = subparsers.add_parser(
        "rollback",
        help="Rollback using a snapshot"
    )
    rollback_parser.add_argument(
        "snapshot_id",
        help="Snapshot ID to rollback to"
    )
    rollback_parser.add_argument(
        "--trigger",
        choices=["manual", "security_violation", "test_failure", "performance_degradation", "error_rate_spike"],
        help="Rollback trigger type"
    )
    rollback_parser.add_argument(
        "--user",
        help="User triggering the rollback"
    )
    rollback_parser.add_argument(
        "--reason",
        help="Reason for rollback"
    )
    rollback_parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    rollback_parser.set_defaults(func=cmd_rollback)
    
    # undo command (quick rollback)
    undo_parser = subparsers.add_parser(
        "undo",
        help="Quick undo for a specific context"
    )
    undo_parser.add_argument(
        "context_id",
        help="Context ID to undo"
    )
    undo_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation prompt"
    )
    undo_parser.add_argument(
        "--user",
        help="User performing the undo"
    )
    undo_parser.set_defaults(func=cmd_undo)
    
    # list-snapshots command
    list_snapshots_parser = subparsers.add_parser(
        "list-snapshots",
        help="List available snapshots"
    )
    list_snapshots_parser.add_argument(
        "--context-id", "-c",
        help="Filter by context ID"
    )
    list_snapshots_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=20,
        help="Maximum number of snapshots to show"
    )
    list_snapshots_parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    list_snapshots_parser.set_defaults(func=cmd_list_snapshots)
    
    # list-rollbacks command
    list_rollbacks_parser = subparsers.add_parser(
        "list-rollbacks",
        help="List rollback operations"
    )
    list_rollbacks_parser.add_argument(
        "--context-id", "-c",
        help="Filter by context ID"
    )
    list_rollbacks_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=20,
        help="Maximum number of rollbacks to show"
    )
    list_rollbacks_parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    list_rollbacks_parser.set_defaults(func=cmd_list_rollbacks)
    
    # rollback-status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show rollback system status"
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    status_parser.set_defaults(func=cmd_rollback_status)
    
    # cleanup-snapshots command
    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="Clean up old snapshots"
    )
    cleanup_parser.add_argument(
        "--retention-days", "-r",
        type=int,
        default=30,
        help="Number of days to retain snapshots"
    )
    cleanup_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation prompt"
    )
    cleanup_parser.set_defaults(func=cmd_cleanup_snapshots)
    
    # monitor command
    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Monitor for issues and auto-rollback if needed"
    )
    monitor_parser.add_argument(
        "context_id",
        help="Context ID to monitor"
    )
    monitor_parser.add_argument(
        "snapshot_id",
        help="Snapshot ID to rollback to if issues detected"
    )
    monitor_parser.add_argument(
        "--duration", "-d",
        type=int,
        default=5,
        help="Monitoring duration in minutes"
    )
    monitor_parser.set_defaults(func=cmd_monitor)
    
    return parser


def main() -> None:
    """Main entry point for the rollback CLI."""
    parser = create_rollback_cli_parser()
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()