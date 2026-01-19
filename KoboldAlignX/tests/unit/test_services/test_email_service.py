"""
Unit tests for email_service.py
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
import smtplib

from src.services.email_service import EmailService, HTML_TEMPLATE


class TestEmailService:
    """Test suite for EmailService."""
    
    def test_initialization(self):
        """Test EmailService initialization."""
        service = EmailService()
        
        assert service.smtp_host is not None
        assert service.smtp_port is not None
        assert service.from_email is not None
        assert service.username is not None
        assert service.password is not None
    
    @patch('src.services.email_service.smtplib.SMTP')
    def test_send_email_success(self, mock_smtp, temp_csv_file):
        """Test successful email sending."""
        # Setup mock SMTP
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # Test
        service = EmailService()
        service.send_email(
            subject="Test Subject",
            html_content="<html><body>Test</body></html>",
            attachment_path=temp_csv_file,
            to_email="test@example.com"
        )
        
        # Verify
        mock_smtp.assert_called_once()
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.send_message.assert_called_once()
    
    @patch('src.services.email_service.smtplib.SMTP')
    def test_send_email_with_cc(self, mock_smtp, temp_csv_file):
        """Test email sending with CC."""
        # Setup mock SMTP
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # Test
        service = EmailService()
        service.send_email(
            subject="Test Subject",
            html_content="<html><body>Test</body></html>",
            attachment_path=temp_csv_file,
            to_email="test@example.com",
            cc_email="cc@example.com"
        )
        
        # Verify
        mock_server.send_message.assert_called_once()
    
    @patch('src.services.email_service.smtplib.SMTP')
    def test_send_email_attachment_not_found(self, mock_smtp):
        """Test email sending with missing attachment."""
        from src.core.exceptions import EmailAttachmentError
        
        service = EmailService()
        
        with pytest.raises(EmailAttachmentError):
            service.send_email(
                subject="Test",
                html_content="<html><body>Test</body></html>",
                attachment_path="/nonexistent/file.csv",
                to_email="test@example.com"
            )
    
    @patch('src.services.email_service.smtplib.SMTP')
    def test_send_email_smtp_error(self, mock_smtp, temp_csv_file):
        """Test handling of SMTP error."""
        from src.core.exceptions import EmailSendError
        
        # Setup mock to raise SMTP exception
        mock_smtp.return_value.__enter__.side_effect = smtplib.SMTPException("SMTP error")
        
        service = EmailService()
        
        with pytest.raises(EmailSendError):
            service.send_email(
                subject="Test",
                html_content="<html><body>Test</body></html>",
                attachment_path=temp_csv_file,
                to_email="test@example.com"
            )
    
    @patch('src.services.email_service.smtplib.SMTP')
    def test_send_email_authentication_failure(self, mock_smtp, temp_csv_file):
        """Test handling of authentication failure."""
        from src.core.exceptions import EmailSendError
        
        # Setup mock
        mock_server = MagicMock()
        mock_server.login.side_effect = smtplib.SMTPAuthenticationError(535, "Authentication failed")
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        service = EmailService()
        
        with pytest.raises(EmailSendError):
            service.send_email(
                subject="Test",
                html_content="<html><body>Test</body></html>",
                attachment_path=temp_csv_file,
                to_email="test@example.com"
            )
    
    @patch('src.services.email_service.smtplib.SMTP')
    def test_send_job_completion_email(self, mock_smtp, temp_csv_file):
        """Test send_job_completion_email method."""
        # Setup mock SMTP
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # Test
        service = EmailService()
        service.send_job_completion_email(
            to_email="test@example.com",
            job_id="28238",
            guidehawk_job_id="GH-12345",
            uwi="100/01-02-003-04W5",
            attachment_path=temp_csv_file
        )
        
        # Verify
        mock_server.send_message.assert_called_once()
    
    def test_render_job_completion_template(self):
        """Test HTML template rendering."""
        service = EmailService()
        
        html = service._render_job_completion_template(
            subject="FallOff_100/01-02-003-04W5",
            uwi="100/01-02-003-04W5",
            job_id="28238",
            guidehawk_job_id="GH-12345"
        )
        
        # Verify template content
        assert "100/01-02-003-04W5" in html
        assert "28238" in html
        assert "GH-12345" in html
        assert "<html>" in html
        assert "</html>" in html
    
    def test_html_template_format(self):
        """Test HTML template has correct format placeholders."""
        # Template should have these placeholders
        assert "{subject}" in HTML_TEMPLATE
        assert "{uwi}" in HTML_TEMPLATE
        assert "{job_id}" in HTML_TEMPLATE
        assert "{guidehawk_job_id}" in HTML_TEMPLATE
    
    @patch('src.services.email_service.smtplib.SMTP')
    def test_email_message_structure(self, mock_smtp, temp_csv_file):
        """Test that email message is structured correctly."""
        # Setup mock
        mock_server = MagicMock()
        sent_message = None
        
        def capture_message(msg):
            nonlocal sent_message
            sent_message = msg
        
        mock_server.send_message = capture_message
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # Test
        service = EmailService()
        service.send_email(
            subject="Test Subject",
            html_content="<html><body>Test Content</body></html>",
            attachment_path=temp_csv_file,
            to_email="recipient@example.com",
            cc_email="cc@example.com"
        )
        
        # Verify message structure
        assert sent_message is not None
        assert sent_message["Subject"] == "Test Subject"
        assert sent_message["To"] == "recipient@example.com"
        assert sent_message["Cc"] == "cc@example.com"


class TestEmailServiceEdgeCases:
    """Test edge cases and special scenarios."""
    
    @patch('src.services.email_service.smtplib.SMTP')
    def test_large_attachment(self, mock_smtp, tmp_path):
        """Test sending email with large attachment."""
        # Create a larger test file
        large_file = tmp_path / "large_report.csv"
        large_file.write_text("header\n" + "data,data,data\n" * 10000)
        
        # Setup mock
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # Test
        service = EmailService()
        service.send_email(
            subject="Large Attachment Test",
            html_content="<html><body>Test</body></html>",
            attachment_path=str(large_file),
            to_email="test@example.com"
        )
        
        # Verify - should complete successfully
        mock_server.send_message.assert_called_once()
    
    @patch('src.services.email_service.smtplib.SMTP')
    def test_special_characters_in_subject(self, mock_smtp, temp_csv_file):
        """Test email with special characters in subject."""
        # Setup mock
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # Test with special characters
        service = EmailService()
        service.send_email(
            subject="Test: Subject with /special\\chars & symbols!",
            html_content="<html><body>Test</body></html>",
            attachment_path=temp_csv_file,
            to_email="test@example.com"
        )
        
        # Verify
        mock_server.send_message.assert_called_once()
    
    @patch('src.services.email_service.smtplib.SMTP')
    def test_empty_html_content(self, mock_smtp, temp_csv_file):
        """Test email with empty HTML content."""
        # Setup mock
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # Test
        service = EmailService()
        service.send_email(
            subject="Empty Content",
            html_content="",
            attachment_path=temp_csv_file,
            to_email="test@example.com"
        )
        
        # Verify
        mock_server.send_message.assert_called_once()