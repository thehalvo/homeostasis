"""
Tests for the audit monitoring and healing audit functionality.
"""

import datetime
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from modules.monitoring.audit_monitor import (
    AuditEvent, 
    AuditMonitor
)
from modules.monitoring.healing_audit import (
    HealingActivityAuditor
)


class TestAuditEvent(unittest.TestCase):
    """Tests for the AuditEvent class."""
    
    def test_from_json(self):
        """Test creating an AuditEvent from JSON."""
        # Test with a complete event
        event_json = {
            'event_id': 'test-id',
            'timestamp': '2023-01-01T12:00:00Z',
            'event_type': 'test_event',
            'status': 'success',
            'severity': 'info',
            'hostname': 'test-host',
            'user': 'test-user',
            'details': {'key': 'value'},
            'source_ip': '127.0.0.1'
        }
        
        event = AuditEvent.from_json(event_json)
        
        self.assertEqual(event.event_id, 'test-id')
        self.assertEqual(event.event_type, 'test_event')
        self.assertEqual(event.status, 'success')
        self.assertEqual(event.severity, 'info')
        self.assertEqual(event.hostname, 'test-host')
        self.assertEqual(event.user, 'test-user')
        self.assertEqual(event.details, {'key': 'value'})
        self.assertEqual(event.source_ip, '127.0.0.1')
        
        # Test timestamp parsing
        self.assertEqual(event.timestamp.year, 2023)
        self.assertEqual(event.timestamp.month, 1)
        self.assertEqual(event.timestamp.day, 1)
        self.assertEqual(event.timestamp.hour, 12)
        
        # Test with minimal event
        minimal_json = {
            'event_id': 'test-id',
            'event_type': 'test_event'
        }
        
        minimal_event = AuditEvent.from_json(minimal_json)
        
        self.assertEqual(minimal_event.event_id, 'test-id')
        self.assertEqual(minimal_event.event_type, 'test_event')
        self.assertEqual(minimal_event.status, '')
        self.assertEqual(minimal_event.severity, '')
        self.assertEqual(minimal_event.hostname, '')
        self.assertEqual(minimal_event.user, '')
        self.assertEqual(minimal_event.details, {})
        self.assertIsNone(minimal_event.source_ip)
        
        # Test source_ip extraction from details
        details_ip_json = {
            'event_id': 'test-id',
            'event_type': 'test_event',
            'details': {'source_ip': '192.168.1.1'}
        }
        
        details_ip_event = AuditEvent.from_json(details_ip_json)
        
        self.assertEqual(details_ip_event.source_ip, '192.168.1.1')


class TestAuditMonitor(unittest.TestCase):
    """Tests for the AuditMonitor class."""
    
    def setUp(self):
        """Set up tests."""
        # Create a temporary log file
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.log_path = self.temp_file.name
        self.temp_file.close()
        
        # Create audit monitor with test config
        self.monitor = AuditMonitor({'log_file': self.log_path})
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.log_path):
            os.unlink(self.log_path)
    
    def test_ensure_log_file(self):
        """Test log file is created if it doesn't exist."""
        # Remove the log file
        os.unlink(self.log_path)
        
        # Call _ensure_log_file
        self.monitor._ensure_log_file()
        
        # Check that the file was created
        self.assertTrue(os.path.exists(self.log_path))
    
    def test_parse_log_line(self):
        """Test parsing a log line."""
        # Valid log line
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        event_data = {
            'event_id': 'test-id',
            'timestamp': datetime.datetime.now().isoformat() + 'Z',
            'event_type': 'test_event',
            'status': 'success',
            'severity': 'info',
            'hostname': 'test-host',
            'user': 'test-user',
            'details': {'key': 'value'}
        }
        log_line = f'{timestamp} [INFO] {json.dumps(event_data)}'
        
        event = self.monitor._parse_log_line(log_line)
        
        self.assertIsNotNone(event)
        self.assertEqual(event.event_id, 'test-id')
        self.assertEqual(event.event_type, 'test_event')
        
        # Invalid log line
        invalid_line = f'{timestamp} [INFO] Not a JSON'
        invalid_event = self.monitor._parse_log_line(invalid_line)
        
        self.assertIsNone(invalid_event)
    
    def test_read_new_events(self):
        """Test reading new events from the log file."""
        # Write some test events to the log file
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        events = []
        
        with open(self.log_path, 'w') as f:
            for i in range(5):
                event_data = {
                    'event_id': f'test-id-{i}',
                    'timestamp': datetime.datetime.now().isoformat() + 'Z',
                    'event_type': 'test_event',
                    'status': 'success',
                    'severity': 'info',
                    'hostname': 'test-host',
                    'user': 'test-user',
                    'details': {'index': i}
                }
                log_line = f'{timestamp} [INFO] {json.dumps(event_data)}'
                f.write(log_line + '\n')
                events.append(event_data)
        
        # Read the events
        new_events = self.monitor.read_new_events()
        
        # Check that all events were read
        self.assertEqual(len(new_events), 5)
        
        # Check event details
        self.assertEqual(new_events[0].event_id, 'test-id-0')
        self.assertEqual(new_events[4].event_id, 'test-id-4')
        
        # Check that last position was updated
        self.assertGreater(self.monitor.last_position, 0)
        
        # Check that events were added to cache
        self.assertEqual(len(self.monitor.event_cache), 5)
        
        # Add more events and check incremental reading
        with open(self.log_path, 'a') as f:
            for i in range(5, 10):
                event_data = {
                    'event_id': f'test-id-{i}',
                    'timestamp': datetime.datetime.now().isoformat() + 'Z',
                    'event_type': 'test_event',
                    'status': 'success',
                    'severity': 'info',
                    'hostname': 'test-host',
                    'user': 'test-user',
                    'details': {'index': i}
                }
                log_line = f'{timestamp} [INFO] {json.dumps(event_data)}'
                f.write(log_line + '\n')
        
        # Read the new events
        more_events = self.monitor.read_new_events()
        
        # Check that only new events were read
        self.assertEqual(len(more_events), 5)
        
        # Check event details
        self.assertEqual(more_events[0].event_id, 'test-id-5')
        self.assertEqual(more_events[4].event_id, 'test-id-9')
        
        # Check cache size
        self.assertEqual(len(self.monitor.event_cache), 10)


class TestHealingActivityAuditor(unittest.TestCase):
    """Tests for the HealingActivityAuditor class."""
    
    def setUp(self):
        """Set up tests."""
        self.auditor = HealingActivityAuditor({'detailed_logging': True})
        
        # Mock the audit_logger to avoid actual logging
        self.mock_logger = MagicMock()
        self.auditor.audit_logger = self.mock_logger
    
    def test_healing_session(self):
        """Test starting and ending a healing session."""
        # Start a session
        session_id = self.auditor.start_healing_session(
            trigger='test',
            user='test-user',
            details={'test': True}
        )
        
        # Check that session was created
        self.assertIn(session_id, self.auditor.active_sessions)
        session = self.auditor.active_sessions[session_id]
        
        self.assertEqual(session['trigger'], 'test')
        self.assertEqual(session['user'], 'test-user')
        self.assertEqual(session['details'], {'test': True})
        self.assertEqual(session['status'], 'in_progress')
        self.assertEqual(len(session['activities']), 0)
        
        # End the session
        self.auditor.end_healing_session(
            session_id,
            status='completed',
            details={'result': 'success'}
        )
        
        # Check that session was removed (due to cleanup_completed_sessions=True)
        self.assertNotIn(session_id, self.auditor.active_sessions)
    
    def test_log_error_detection(self):
        """Test logging error detection."""
        session_id = self.auditor.start_healing_session()
        
        self.auditor.log_error_detection(
            session_id,
            error_id='test-error',
            error_type='KeyError',
            source='test-source',
            details={'key': 'missing_key'}
        )
        
        # Check that activity was added to session
        session = self.auditor.active_sessions[session_id]
        self.assertEqual(len(session['activities']), 1)
        
        activity = session['activities'][0]
        self.assertEqual(activity['activity_type'], 'error_detection')
        self.assertEqual(activity['error_id'], 'test-error')
        self.assertEqual(activity['error_type'], 'KeyError')
        self.assertEqual(activity['source'], 'test-source')
        self.assertEqual(activity['details'], {'key': 'missing_key'})
    
    def test_get_session_history(self):
        """Test getting session history."""
        session_id = self.auditor.start_healing_session()
        
        # Add some activities
        self.auditor.log_error_detection(
            session_id,
            error_id='test-error',
            error_type='KeyError',
            source='test-source'
        )
        
        self.auditor.log_error_analysis(
            session_id,
            error_id='test-error',
            analysis_type='rule_based',
            root_cause='missing_key',
            confidence=0.95,
            duration_ms=123.45
        )
        
        # Get session history
        history = self.auditor.get_session_history(session_id)
        
        # Check history contents
        self.assertEqual(history['session_id'], session_id)
        self.assertEqual(history['status'], 'in_progress')
        self.assertEqual(len(history['activities']), 2)
        
        # Check activity details
        detection = history['activities'][0]
        analysis = history['activities'][1]
        
        self.assertEqual(detection['activity_type'], 'error_detection')
        self.assertEqual(detection['error_id'], 'test-error')
        
        self.assertEqual(analysis['activity_type'], 'error_analysis')
        self.assertEqual(analysis['error_id'], 'test-error')
        self.assertEqual(analysis['root_cause'], 'missing_key')
        self.assertAlmostEqual(analysis['confidence'], 0.95)
        self.assertAlmostEqual(analysis['duration_ms'], 123.45)


if __name__ == '__main__':
    unittest.main()