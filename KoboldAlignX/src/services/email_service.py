"""
Email service for sending job reports.

This module provides email functionality for sending job completion
reports with CSV attachments via SMTP.
"""
import os
import logging
import smtplib
from email.message import EmailMessage
from email.utils import formataddr
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from config import (
    SMTP_SERVER,
    SMTP_PORT,
    OUTLOOK_FROM_EMAIL,
    OUTLOOK_EMAIL,
    OUTLOOK_PASSWORD
)

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EmailConfig:
    """Email service configuration."""
    smtp_server: str
    smtp_port: int
    from_email: str
    from_name: str
    smtp_username: str
    smtp_password: str
    use_tls: bool = True
    timeout: int = 30
    
    @classmethod
    def from_env(cls) -> 'EmailConfig':
        """Create configuration from environment/config."""
        return cls(
            smtp_server=SMTP_SERVER,
            smtp_port=SMTP_PORT,
            from_email=OUTLOOK_FROM_EMAIL,
            from_name="Kobold Completions Inc.",
            smtp_username=OUTLOOK_EMAIL,
            smtp_password=OUTLOOK_PASSWORD
        )


@dataclass
class JobInfo:
    """Job information for email body."""
    uwi: str
    reliance_job_id: str
    guidehawk_job_id: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for template rendering."""
        return {
            'uwi': self.uwi,
            'reliance_job_id': self.reliance_job_id,
            'guidehawk_job_id': self.guidehawk_job_id
        }


# ============================================================================
# Email Templates
# ============================================================================

class EmailTemplate:
    """Email template generator."""
    
    @staticmethod
    def generate_job_report_html(
        subject: str,
        job_info: JobInfo,
        custom_message: Optional[str] = None
    ) -> str:
        """
        Generate HTML email body for job report.
        
        Args:
            subject: Email subject line
            job_info: Job information
            custom_message: Optional custom message to include
            
        Returns:
            HTML string
        """
        message_section = EmailTemplate._generate_message_section(custom_message)
        job_details = EmailTemplate._generate_job_details(job_info)
        
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{subject}</title>
    <style>
        {EmailTemplate._get_styles()}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">{subject}</div>
        <div class="content">
            <p>Hello,</p>
            {message_section}
            {job_details}
            <p>
                If you have any questions, feel free to reach out to our support team.
            </p>
            <p>Best regards,</p>
        </div>
        <div class="footer">
            {EmailTemplate._get_footer_disclaimer()}
        </div>
    </div>
</body>
</html>
"""
    
    @staticmethod
    def _generate_message_section(custom_message: Optional[str]) -> str:
        """Generate the main message section."""
        if custom_message:
            return f"<p>{custom_message}</p>"
        
        return """
            <p>
                We are pleased to inform you that the latest processed data file is now available. 
                Please find the attached document for your review.
            </p>
"""
    
    @staticmethod
    def _generate_job_details(job_info: JobInfo) -> str:
        """Generate job details section."""
        return f"""
            <p>
                <b>UWI:</b> {job_info.uwi}<br>
                <b>Reliance Job ID:</b> {job_info.reliance_job_id}<br>
                <b>Kobold Job ID:</b> {job_info.guidehawk_job_id}
            </p>
"""
    
    @staticmethod
    def _get_styles() -> str:
        """Get CSS styles for email."""
        return """
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #ffffff;
        }
        .container {
            width: 100%;
            background-color: #ffffff;
            padding: 20px;
        }
        .header {
            font-size: 24px;
            font-weight: bold;
            color: #333333;
            margin-bottom: 20px;
        }
        .content {
            font-size: 16px;
            line-height: 1.6;
            color: #555555;
        }
        .content p {
            margin: 10px 0;
        }
        .footer {
            font-size: 12px;
            color: #777777;
            padding: 20px;
            margin-top: 20px;
            text-align: left;
            border-top: 1px solid #eeeeee;
        }
"""
    
    @staticmethod
    def _get_footer_disclaimer() -> str:
        """Get footer disclaimer text."""
        return """
            <p>
                This e-mail message is intended only for the above named recipient(s) and may contain 
                information that is privileged, confidential and/or exempt from disclosure under applicable law. 
                If you have received this message in error or are not the named recipient(s), please immediately 
                notify the sender, delete this e-mail message without making a copy and do not disclose or relay 
                this e-mail message to anyone.
            </p>
"""
    
    @staticmethod
    def generate_plain_text() -> str:
        """Generate plain text fallback."""
        return "This is an HTML email. Please view it in an HTML-compatible email client."

class EmailBuilder:
    """Builder for constructing email messages."""
    
    def __init__(self, config: EmailConfig):
        self.config = config
        self.msg = EmailMessage()
    
    def set_subject(self, subject: str) -> 'EmailBuilder':
        """Set email subject."""
        self.msg["Subject"] = subject
        return self
    
    def set_from(self, name: Optional[str] = None, email: Optional[str] = None) -> 'EmailBuilder':
        """Set sender information."""
        from_name = name or self.config.from_name
        from_email = email or self.config.from_email
        self.msg["From"] = formataddr((from_name, from_email))
        return self
    
    def set_to(self, recipients: List[str]) -> 'EmailBuilder':
        if isinstance(recipients, str):
            recipients = [recipients]
        self.msg["To"] = ", ".join(recipients)
        return self
    
    def set_cc(self, recipients: Optional[List[str]]) -> 'EmailBuilder':
        if recipients:
            if isinstance(recipients, str):
                recipients = [recipients]
            self.msg["Cc"] = ", ".join(recipients)
        return self
    
    def set_bcc(self, recipients: Optional[List[str]]) -> 'EmailBuilder':
        if recipients:
            if isinstance(recipients, str):
                recipients = [recipients]
            self.msg["Bcc"] = ", ".join(recipients)
        return self
    
    def set_body(self, html_content: str, plain_text: Optional[str] = None) -> 'EmailBuilder':
        # Set plain text version
        plain = plain_text or EmailTemplate.generate_plain_text()
        self.msg.set_content(plain)
        
        # Add HTML version
        self.msg.add_alternative(html_content, subtype="html")
        return self
    
    def add_attachment(
        self,
        file_path: str,
        filename: Optional[str] = None,
        maintype: str = "application",
        subtype: str = "octet-stream"
    ) -> 'EmailBuilder':
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Attachment file not found: {file_path}")
        
        with open(path, "rb") as file:
            logger.debug(f"Attaching file: {file_path}")
            self.msg.add_attachment(
                file.read(),
                maintype=maintype,
                subtype=subtype,
                filename=filename or path.name
            )
        
        return self
    
    def build(self) -> EmailMessage:
        """Build and return the email message."""
        return self.msg


# ============================================================================
# SMTP Client
# ============================================================================

class SMTPClient:
    """SMTP client for sending emails."""
    
    def __init__(self, config: EmailConfig):        
        self.config = config
    
    def send(self, message: EmailMessage) -> None:
        logger.info(f"Connecting to SMTP server: {self.config.smtp_server}:{self.config.smtp_port}")
        
        with smtplib.SMTP(
            self.config.smtp_server,
            self.config.smtp_port,
            timeout=self.config.timeout
        ) as smtp:
            # Enable TLS if configured
            if self.config.use_tls:
                logger.debug("Starting TLS...")
                smtp.starttls()
            
            # Authenticate
            logger.debug("Authenticating...")
            smtp.login(self.config.smtp_username, self.config.smtp_password)
            
            # Send message
            logger.debug("Sending message...")
            smtp.send_message(message)
            
            logger.info("Email sent successfully")

class EmailService:
    """
    Service for sending email notifications with attachments.
    
    This class provides a high-level interface for sending job reports
    and other notifications via email.
    """
    
    def __init__(self, config: Optional[EmailConfig] = None):
        self.config = config or EmailConfig.from_env()
        self.smtp_client = SMTPClient(self.config)
    
    def send_report(
        self,
        subject: str,
        uwi: str,
        reliance_job_id: str,
        guidehawk_job_id: str,
        attachment_path: str,
        to_email: str,
        cc_email: Optional[str] = None,
        custom_message: Optional[str] = None
    ) -> bool:
        """ Send a job report email with CSV attachment. """
        try:
            # Create job info
            job_info = JobInfo(
                uwi=uwi,
                reliance_job_id=reliance_job_id,
                guidehawk_job_id=guidehawk_job_id
            )
            
            # Generate HTML body
            html_content = EmailTemplate.generate_job_report_html(
                subject=subject,
                job_info=job_info,
                custom_message=custom_message
            )
            
            # Build email
            message = (EmailBuilder(self.config)
                .set_subject(subject)
                .set_from()
                .set_to([to_email])
                .set_cc([cc_email] if cc_email else None)
                .set_body(html_content)
                .add_attachment(attachment_path)
                .build())
            
            # Send email
            self.smtp_client.send(message)
            
            logger.info(f"Job report sent to {to_email}")
            return True
            
        except FileNotFoundError as e:
            logger.error(f"Attachment file not found: {str(e)}")
            return False
        except IOError as e:
            logger.error(f"I/O error sending email: {str(e)}")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error sending email: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending email: {str(e)}")
            return False
    
    def send_custom_email(
        self,
        subject: str,
        to_emails: List[str],
        html_body: str,
        attachments: Optional[List[str]] = None,
        cc_emails: Optional[List[str]] = None,
        bcc_emails: Optional[List[str]] = None,
        plain_text_body: Optional[str] = None
    ) -> bool:
        """ Send a custom email with full control over content. """
        try:
            # Build email
            builder = (EmailBuilder(self.config)
                .set_subject(subject)
                .set_from()
                .set_to(to_emails)
                .set_cc(cc_emails)
                .set_bcc(bcc_emails)
                .set_body(html_body, plain_text_body))
            
            # Add attachments if provided
            if attachments:
                for attachment_path in attachments:
                    builder.add_attachment(attachment_path)
            
            message = builder.build()
            
            # Send email
            self.smtp_client.send(message)
            
            logger.info(f"Custom email sent to {len(to_emails)} recipient(s)")
            return True
            
        except FileNotFoundError as e:
            logger.error(f"Attachment file not found: {str(e)}")
            return False
        except IOError as e:
            logger.error(f"I/O error sending email: {str(e)}")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error sending email: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending email: {str(e)}")
            return False
    
    def send_notification(
        self,
        subject: str,
        message: str,
        to_emails: List[str],
        cc_emails: Optional[List[str]] = None
    ) -> bool:
        """
        Send a simple text notification email.
        
        Args:
            subject: Email subject
            message: Notification message
            to_emails: List of recipient email addresses
            cc_emails: Optional list of CC email addresses
            
        Returns:
            True if email sent successfully, False otherwise
        """
        # Generate simple HTML
        html_body = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{subject}</title>
</head>
<body style="font-family: Arial, sans-serif; padding: 20px;">
    <h2>{subject}</h2>
    <p>{message}</p>
</body>
</html>
"""
        
        return self.send_custom_email(
            subject=subject,
            to_emails=to_emails,
            html_body=html_body,
            cc_emails=cc_emails,
            plain_text_body=message
        )
    
    def test_connection(self) -> bool:
        """ Test SMTP connection and authentication."""
        try:
            logger.info("Testing SMTP connection...")
            
            with smtplib.SMTP(
                self.config.smtp_server,
                self.config.smtp_port,
                timeout=self.config.timeout
            ) as smtp:
                if self.config.use_tls:
                    smtp.starttls()
                
                smtp.login(self.config.smtp_username, self.config.smtp_password)
                
                logger.info("SMTP connection test successful")
                return True
                
        except smtplib.SMTPException as e:
            logger.error(f"SMTP connection test failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error testing connection: {str(e)}")
            return False


def create_email_service(config: Optional[EmailConfig] = None) -> EmailService:
    """ Create an email service instance. """
    return EmailService(config)


def send_job_report(
    subject: str,
    uwi: str,
    reliance_job_id: str,
    guidehawk_job_id: str,
    attachment_path: str,
    to_email: str,
    cc_email: Optional[str] = None
) -> bool:
    """ Convenience function to send a job report. """
    service = EmailService()
    return service.send_report(
        subject=subject,
        uwi=uwi,
        reliance_job_id=reliance_job_id,
        guidehawk_job_id=guidehawk_job_id,
        attachment_path=attachment_path,
        to_email=to_email,
        cc_email=cc_email
    )