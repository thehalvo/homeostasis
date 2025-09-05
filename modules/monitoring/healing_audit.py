"""
Healing Activity Audit Logging for Homeostasis.

This module focuses specifically on comprehensive audit logging for all healing 
activities, from detection through analysis, patch generation, testing, and deployment.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from modules.security.audit import get_audit_logger, log_event, log_fix

logger = logging.getLogger(__name__)


class HealingActivityAuditor:
    """
    Specialized auditor for healing activities that ensures comprehensive
    tracking and logging of all steps in the self-healing process.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the healing activity auditor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.audit_logger = get_audit_logger()
        
        # Extract config settings
        self.enabled = self.config.get('enabled', True)
        self.detailed_logging = self.config.get('detailed_logging', True)
        self.track_execution_time = self.config.get('track_execution_time', True)
        
        # Track healing sessions
        self.active_sessions = {}
    
    def start_healing_session(self, trigger: str = 'scheduled', user: str = 'system',
                             details: Dict[str, Any] = None) -> str:
        """
        Start a new healing session.
        
        Args:
            trigger: What triggered the healing session
            user: User who initiated the session
            details: Additional details about the session
            
        Returns:
            str: Session ID
        """
        if not self.enabled:
            return str(uuid.uuid4())
            
        session_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        # Create session record
        session = {
            'session_id': session_id,
            'start_time': timestamp,
            'trigger': trigger,
            'user': user,
            'details': details or {},
            'activities': [],
            'end_time': None,
            'status': 'in_progress'
        }
        
        # Store in active sessions
        self.active_sessions[session_id] = session
        
        # Log session start
        log_event(
            event_type='healing_session_started',
            user=user,
            details={
                'session_id': session_id,
                'trigger': trigger,
                **(details or {})
            }
        )
        
        return session_id
    
    def end_healing_session(self, session_id: str, status: str = 'completed',
                           details: Dict[str, Any] = None) -> None:
        """
        End a healing session.
        
        Args:
            session_id: Session ID
            status: Final status of the session
            details: Additional details about the session
        """
        if not self.enabled or session_id not in self.active_sessions:
            return
            
        # Update session record
        session = self.active_sessions[session_id]
        session['end_time'] = datetime.utcnow()
        session['status'] = status
        
        # Calculate duration
        duration = (session['end_time'] - session['start_time']).total_seconds()
        
        # Log session end
        log_event(
            event_type='healing_session_ended',
            user=session['user'],
            details={
                'session_id': session_id,
                'status': status,
                'duration': duration,
                'activities': len(session['activities']),
                **(details or {})
            }
        )
        
        # Remove from active sessions if cleanup is enabled
        if self.config.get('cleanup_completed_sessions', True):
            del self.active_sessions[session_id]
    
    def log_error_detection(self, session_id: str, error_id: str, 
                          error_type: str, source: str,
                          details: Dict[str, Any] = None) -> None:
        """
        Log an error detection activity.
        
        Args:
            session_id: Session ID
            error_id: Error ID
            error_type: Type of error detected
            source: Source of the error (log file, monitoring, etc.)
            details: Additional details about the error
        """
        if not self.enabled:
            return
            
        activity = {
            'activity_type': 'error_detection',
            'timestamp': datetime.utcnow(),
            'error_id': error_id,
            'error_type': error_type,
            'source': source,
            'details': details or {}
        }
        
        # Add to session if exists
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['activities'].append(activity)
        
        # Log the activity
        log_event(
            event_type='error_detected',
            details={
                'session_id': session_id,
                'error_id': error_id,
                'error_type': error_type,
                'source': source,
                **(details or {})
            }
        )
    
    def log_error_analysis(self, session_id: str, error_id: str,
                         analysis_type: str, root_cause: str, 
                         confidence: float, 
                         duration_ms: Optional[float] = None,
                         details: Dict[str, Any] = None) -> None:
        """
        Log an error analysis activity.
        
        Args:
            session_id: Session ID
            error_id: Error ID
            analysis_type: Type of analysis (rule_based, ai_based, etc.)
            root_cause: Identified root cause
            confidence: Confidence score (0.0-1.0)
            duration_ms: Duration in milliseconds
            details: Additional details about the analysis
        """
        if not self.enabled:
            return
            
        activity = {
            'activity_type': 'error_analysis',
            'timestamp': datetime.utcnow(),
            'error_id': error_id,
            'analysis_type': analysis_type,
            'root_cause': root_cause,
            'confidence': confidence
        }
        
        if duration_ms is not None and self.track_execution_time:
            activity['duration_ms'] = duration_ms
            
        if details:
            activity['details'] = details
        
        # Add to session if exists
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['activities'].append(activity)
        
        # Log the activity
        log_event(
            event_type='error_analyzed',
            details={
                'session_id': session_id,
                'error_id': error_id,
                'analysis_type': analysis_type,
                'root_cause': root_cause,
                'confidence': confidence,
                **(details or {}),
                **({'duration_ms': duration_ms} if duration_ms is not None else {})
            }
        )
    
    def log_patch_generation(self, session_id: str, error_id: str,
                           fix_id: str, fix_type: str, file_path: str,
                           template_name: Optional[str] = None,
                           confidence: float = 0.0,
                           details: Dict[str, Any] = None) -> None:
        """
        Log a patch generation activity.
        
        Args:
            session_id: Session ID
            error_id: Error ID
            fix_id: Fix ID
            fix_type: Type of fix
            file_path: Path to the file being fixed
            template_name: Name of the template used
            confidence: Confidence score (0.0-1.0)
            details: Additional details about the patch
        """
        if not self.enabled:
            return
            
        activity = {
            'activity_type': 'patch_generation',
            'timestamp': datetime.utcnow(),
            'error_id': error_id,
            'fix_id': fix_id,
            'fix_type': fix_type,
            'file_path': file_path,
            'confidence': confidence
        }
        
        if template_name:
            activity['template_name'] = template_name
            
        if details:
            activity['details'] = details
        
        # Add to session if exists
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['activities'].append(activity)
        
        # Log the activity
        log_fix(
            fix_id=fix_id,
            event_type='fix_generated',
            details={
                'session_id': session_id,
                'error_id': error_id,
                'fix_type': fix_type,
                'file_path': file_path,
                'confidence': confidence,
                **({'template_name': template_name} if template_name else {}),
                **(details or {})
            }
        )
    
    def log_patch_application(self, session_id: str, fix_id: str,
                            file_path: str, status: str = 'success',
                            details: Dict[str, Any] = None) -> None:
        """
        Log a patch application activity.
        
        Args:
            session_id: Session ID
            fix_id: Fix ID
            file_path: Path to the file being fixed
            status: Status of the application
            details: Additional details about the application
        """
        if not self.enabled:
            return
            
        activity = {
            'activity_type': 'patch_application',
            'timestamp': datetime.utcnow(),
            'fix_id': fix_id,
            'file_path': file_path,
            'status': status
        }
        
        if details:
            activity['details'] = details
        
        # Add to session if exists
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['activities'].append(activity)
        
        # Log the activity
        log_fix(
            fix_id=fix_id,
            event_type='fix_applied',
            status=status,
            details={
                'session_id': session_id,
                'file_path': file_path,
                **(details or {})
            }
        )
    
    def log_test_execution(self, session_id: str, fix_id: str,
                          test_type: str, status: str,
                          duration_ms: Optional[float] = None,
                          details: Dict[str, Any] = None) -> None:
        """
        Log a test execution activity.
        
        Args:
            session_id: Session ID
            fix_id: Fix ID
            test_type: Type of test (unit, integration, system)
            status: Status of the test
            duration_ms: Duration in milliseconds
            details: Additional details about the test
        """
        if not self.enabled:
            return
            
        activity = {
            'activity_type': 'test_execution',
            'timestamp': datetime.utcnow(),
            'fix_id': fix_id,
            'test_type': test_type,
            'status': status
        }
        
        if duration_ms is not None and self.track_execution_time:
            activity['duration_ms'] = duration_ms
            
        if details:
            activity['details'] = details
        
        # Add to session if exists
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['activities'].append(activity)
        
        # Log the activity
        log_fix(
            fix_id=fix_id,
            event_type='fix_tested',
            status=status,
            details={
                'session_id': session_id,
                'test_type': test_type,
                **({'duration_ms': duration_ms} if duration_ms is not None else {}),
                **(details or {})
            }
        )
    
    def log_deployment(self, session_id: str, fix_id: str,
                      environment: str, status: str,
                      deployment_type: str = 'standard',
                      user: str = 'system',
                      details: Dict[str, Any] = None) -> None:
        """
        Log a deployment activity.
        
        Args:
            session_id: Session ID
            fix_id: Fix ID
            environment: Environment being deployed to
            status: Status of the deployment
            deployment_type: Type of deployment (standard, canary, etc.)
            user: User performing the deployment
            details: Additional details about the deployment
        """
        if not self.enabled:
            return
            
        activity = {
            'activity_type': 'deployment',
            'timestamp': datetime.utcnow(),
            'fix_id': fix_id,
            'environment': environment,
            'status': status,
            'deployment_type': deployment_type,
            'user': user
        }
        
        if details:
            activity['details'] = details
        
        # Add to session if exists
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['activities'].append(activity)
        
        # Log the activity
        log_fix(
            fix_id=fix_id,
            event_type='fix_deployed',
            user=user,
            status=status,
            details={
                'session_id': session_id,
                'environment': environment,
                'deployment_type': deployment_type,
                **(details or {})
            }
        )
    
    def log_canary_update(self, session_id: str, fix_id: str,
                        percentage: float, status: str,
                        metrics: Dict[str, float] = None,
                        user: str = 'system',
                        details: Dict[str, Any] = None) -> None:
        """
        Log a canary deployment update.
        
        Args:
            session_id: Session ID
            fix_id: Fix ID
            percentage: Traffic percentage
            status: Status of the canary
            metrics: Performance metrics
            user: User performing the canary update
            details: Additional details
        """
        if not self.enabled:
            return
            
        activity = {
            'activity_type': 'canary_update',
            'timestamp': datetime.utcnow(),
            'fix_id': fix_id,
            'percentage': percentage,
            'status': status,
            'user': user
        }
        
        if metrics:
            activity['metrics'] = metrics
            
        if details:
            activity['details'] = details
        
        # Add to session if exists
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['activities'].append(activity)
        
        # Log the activity
        log_fix(
            fix_id=fix_id,
            event_type='canary_update',
            user=user,
            status=status,
            details={
                'session_id': session_id,
                'percentage': percentage,
                **({'metrics': metrics} if metrics else {}),
                **(details or {})
            }
        )
    
    def log_human_interaction(self, session_id: str, fix_id: str,
                            interaction_type: str, user: str,
                            details: Dict[str, Any] = None) -> None:
        """
        Log a human interaction with the healing process.
        
        Args:
            session_id: Session ID
            fix_id: Fix ID
            interaction_type: Type of interaction (e.g., review, approval, rejection)
            user: User interacting with the system
            details: Additional details about the interaction
        """
        if not self.enabled:
            return
            
        activity = {
            'activity_type': 'human_interaction',
            'timestamp': datetime.utcnow(),
            'fix_id': fix_id,
            'interaction_type': interaction_type,
            'user': user
        }
        
        if details:
            activity['details'] = details
        
        # Add to session if exists
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['activities'].append(activity)
        
        # Map interaction type to event type
        event_type_map = {
            'review': 'fix_reviewed',
            'approval': 'fix_approved',
            'rejection': 'fix_rejected',
            'modification': 'fix_modified',
            'comment': 'fix_commented'
        }
        
        event_type = event_type_map.get(interaction_type, f'fix_{interaction_type}')
        
        # Log the activity
        log_fix(
            fix_id=fix_id,
            event_type=event_type,
            user=user,
            details={
                'session_id': session_id,
                **(details or {})
            }
        )
    
    def log_rollback(self, session_id: str, fix_id: str,
                   status: str, reason: str,
                   user: str = 'system',
                   details: Dict[str, Any] = None) -> None:
        """
        Log a rollback activity.
        
        Args:
            session_id: Session ID
            fix_id: Fix ID
            status: Status of the rollback
            reason: Reason for the rollback
            user: User performing the rollback
            details: Additional details about the rollback
        """
        if not self.enabled:
            return
            
        activity = {
            'activity_type': 'rollback',
            'timestamp': datetime.utcnow(),
            'fix_id': fix_id,
            'status': status,
            'reason': reason,
            'user': user
        }
        
        if details:
            activity['details'] = details
        
        # Add to session if exists
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['activities'].append(activity)
        
        # Log the activity
        log_fix(
            fix_id=fix_id,
            event_type='fix_rolled_back',
            user=user,
            status=status,
            details={
                'session_id': session_id,
                'reason': reason,
                **(details or {})
            }
        )
    
    def get_session_history(self, session_id: str) -> Dict[str, Any]:
        """
        Get the history of a healing session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dict[str, Any]: Session history or empty dict if not found
        """
        return self.active_sessions.get(session_id, {})
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """
        Get all active healing sessions.
        
        Returns:
            List[Dict[str, Any]]: List of active sessions
        """
        return list(self.active_sessions.values())


# Singleton instance for app-wide use
_healing_auditor = None


def get_healing_auditor(config: Dict[str, Any] = None) -> HealingActivityAuditor:
    """
    Get or create the singleton HealingActivityAuditor instance.
    
    Args:
        config: Optional configuration for the auditor
        
    Returns:
        HealingActivityAuditor: The healing activity auditor instance
    """
    global _healing_auditor
    if _healing_auditor is None:
        _healing_auditor = HealingActivityAuditor(config)
    return _healing_auditor


def start_healing_session(trigger: str = 'scheduled', user: str = 'system',
                         details: Dict[str, Any] = None) -> str:
    """
    Start a new healing session.
    
    Args:
        trigger: What triggered the healing session
        user: User who initiated the session
        details: Additional details about the session
        
    Returns:
        str: Session ID
    """
    return get_healing_auditor().start_healing_session(trigger, user, details)


def end_healing_session(session_id: str, status: str = 'completed',
                       details: Dict[str, Any] = None) -> None:
    """
    End a healing session.
    
    Args:
        session_id: Session ID
        status: Final status of the session
        details: Additional details about the session
    """
    get_healing_auditor().end_healing_session(session_id, status, details)


def log_error_detection(session_id: str, error_id: str, 
                       error_type: str, source: str,
                       details: Dict[str, Any] = None) -> None:
    """
    Log an error detection activity.
    
    Args:
        session_id: Session ID
        error_id: Error ID
        error_type: Type of error detected
        source: Source of the error (log file, monitoring, etc.)
        details: Additional details about the error
    """
    get_healing_auditor().log_error_detection(session_id, error_id, 
                                             error_type, source, details)


def log_error_analysis(session_id: str, error_id: str,
                      analysis_type: str, root_cause: str, 
                      confidence: float, 
                      duration_ms: Optional[float] = None,
                      details: Dict[str, Any] = None) -> None:
    """
    Log an error analysis activity.
    
    Args:
        session_id: Session ID
        error_id: Error ID
        analysis_type: Type of analysis (rule_based, ai_based, etc.)
        root_cause: Identified root cause
        confidence: Confidence score (0.0-1.0)
        duration_ms: Duration in milliseconds
        details: Additional details about the analysis
    """
    get_healing_auditor().log_error_analysis(session_id, error_id,
                                           analysis_type, root_cause,
                                           confidence, duration_ms, details)


def log_patch_generation(session_id: str, error_id: str,
                       fix_id: str, fix_type: str, file_path: str,
                       template_name: Optional[str] = None,
                       confidence: float = 0.0,
                       details: Dict[str, Any] = None) -> None:
    """
    Log a patch generation activity.
    
    Args:
        session_id: Session ID
        error_id: Error ID
        fix_id: Fix ID
        fix_type: Type of fix
        file_path: Path to the file being fixed
        template_name: Name of the template used
        confidence: Confidence score (0.0-1.0)
        details: Additional details about the patch
    """
    get_healing_auditor().log_patch_generation(session_id, error_id, fix_id,
                                            fix_type, file_path, template_name,
                                            confidence, details)


def log_patch_application(session_id: str, fix_id: str,
                        file_path: str, status: str = 'success',
                        details: Dict[str, Any] = None) -> None:
    """
    Log a patch application activity.
    
    Args:
        session_id: Session ID
        fix_id: Fix ID
        file_path: Path to the file being fixed
        status: Status of the application
        details: Additional details about the application
    """
    get_healing_auditor().log_patch_application(session_id, fix_id,
                                              file_path, status, details)


def log_test_execution(session_id: str, fix_id: str,
                      test_type: str, status: str,
                      duration_ms: Optional[float] = None,
                      details: Dict[str, Any] = None) -> None:
    """
    Log a test execution activity.
    
    Args:
        session_id: Session ID
        fix_id: Fix ID
        test_type: Type of test (unit, integration, system)
        status: Status of the test
        duration_ms: Duration in milliseconds
        details: Additional details about the test
    """
    get_healing_auditor().log_test_execution(session_id, fix_id,
                                           test_type, status,
                                           duration_ms, details)


def log_deployment(session_id: str, fix_id: str,
                  environment: str, status: str,
                  deployment_type: str = 'standard',
                  user: str = 'system',
                  details: Dict[str, Any] = None) -> None:
    """
    Log a deployment activity.
    
    Args:
        session_id: Session ID
        fix_id: Fix ID
        environment: Environment being deployed to
        status: Status of the deployment
        deployment_type: Type of deployment (standard, canary, etc.)
        user: User performing the deployment
        details: Additional details about the deployment
    """
    get_healing_auditor().log_deployment(session_id, fix_id,
                                       environment, status,
                                       deployment_type, user, details)


def log_human_interaction(session_id: str, fix_id: str,
                        interaction_type: str, user: str,
                        details: Dict[str, Any] = None) -> None:
    """
    Log a human interaction with the healing process.
    
    Args:
        session_id: Session ID
        fix_id: Fix ID
        interaction_type: Type of interaction (e.g., review, approval, rejection)
        user: User interacting with the system
        details: Additional details about the interaction
    """
    get_healing_auditor().log_human_interaction(session_id, fix_id,
                                              interaction_type, user, details)