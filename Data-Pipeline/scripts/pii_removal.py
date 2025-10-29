"""
PII Detection and Removal Module
Task 2: Data Preprocessing & Cleaning

This module detects and removes Personally Identifiable Information (PII)
from code files including emails, IP addresses, API keys, and secrets.
"""

import re
import logging
from typing import Dict, List, Tuple, Set
import hashlib
from pathlib import Path

class PIIRemover:
    def __init__(self, log_removed_items: bool = True):
        """
        Initialize PII removal with configurable logging.
        
        Args:
            log_removed_items: Whether to log what PII items were removed
        """
        self.log_removed_items = log_removed_items
        self.logger = logging.getLogger(__name__)
        
        # Compile regex patterns for performance
        self._compile_patterns()
        
        # Statistics tracking
        self.stats = {
            'files_processed': 0,
            'pii_items_removed': 0,
            'emails_removed': 0,
            'ips_removed': 0,
            'api_keys_removed': 0,
            'secrets_removed': 0,
            'urls_removed': 0
        }
    
    def _compile_patterns(self):
        """Compile all regex patterns for better performance"""
        
        # Email patterns - more comprehensive
        self.email_patterns = [
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            re.compile(r'\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\s*\.\s*[A-Z|a-z]{2,}\b'),
        ]
        
        # IP Address patterns
        self.ip_patterns = [
            # IPv4
            re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            # IPv6 (simplified)
            re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'),
            re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b'),
        ]
        
        # API Key patterns (common formats)
        self.api_key_patterns = [
            # Generic API key patterns
            re.compile(r'(?i)(api[_-]?key|apikey|api_key|access[_-]?key|accesskey)\s*[=:]\s*["\']?([A-Za-z0-9_\-]{16,})["\']?'),
            re.compile(r'(?i)(secret[_-]?key|secretkey|secret_key)\s*[=:]\s*["\']?([A-Za-z0-9_\-]{16,})["\']?'),
            re.compile(r'(?i)(token)\s*[=:]\s*["\']?([A-Za-z0-9_\-\.]{20,})["\']?'),
            
            # AWS keys
            re.compile(r'(?i)(aws[_-]?access[_-]?key[_-]?id|aws_access_key_id)\s*[=:]\s*["\']?([A-Z0-9]{20})["\']?'),
            re.compile(r'(?i)(aws[_-]?secret[_-]?access[_-]?key|aws_secret_access_key)\s*[=:]\s*["\']?([A-Za-z0-9/+=]{40})["\']?'),
            
            # GitHub tokens
            re.compile(r'(?i)(github[_-]?token|gh[_-]?token)\s*[=:]\s*["\']?(ghp_[A-Za-z0-9_]{36})["\']?'),
            re.compile(r'(?i)(github[_-]?pat|personal[_-]?access[_-]?token)\s*[=:]\s*["\']?([A-Za-z0-9_]{40})["\']?'),
            
            # Google API keys
            re.compile(r'(?i)(google[_-]?api[_-]?key)\s*[=:]\s*["\']?(AIza[0-9A-Za-z_\-]{35})["\']?'),
            
            # Slack tokens
            re.compile(r'(?i)(slack[_-]?token)\s*[=:]\s*["\']?(xox[bpoa]-[0-9]{12}-[0-9]{12}-[0-9A-Za-z]{24})["\']?'),
            
            # Generic high-entropy strings that look like secrets
            re.compile(r'(?i)(password|passwd|pwd|secret)\s*[=:]\s*["\']?([A-Za-z0-9!@#$%^&*()_+\-=\[\]{}|;:,.<>?]{8,})["\']?'),
        ]

        # Standalone tokens (e.g., GitHub PATs appearing in comments or text)
        self.standalone_token_patterns = [
            re.compile(r'(ghp_[A-Za-z0-9_]{20,})'),
        ]
        
        # URL patterns (to remove sensitive URLs)
        self.url_patterns = [
            re.compile(r'https?://[A-Za-z0-9.-]+(?:/[A-Za-z0-9._~:/?#[\]@!$&\'()*+,;=-]*)?'),
            re.compile(r'ftp://[A-Za-z0-9.-]+(?:/[A-Za-z0-9._~:/?#[\]@!$&\'()*+,;=-]*)?'),
        ]
        
        # Database connection strings
        self.db_patterns = [
            re.compile(r'(?i)(mongodb|mysql|postgresql|postgres)://[^\s\'"]+'),
            re.compile(r'(?i)(host|server)\s*[=:]\s*["\']?([A-Za-z0-9.-]+)["\']?.*(user|username|uid)\s*[=:]\s*["\']?([A-Za-z0-9_]+)["\']?'),
        ]
        
        # Credit card patterns (basic)
        self.cc_patterns = [
            re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),  # 16 digit cards with separators
            re.compile(r'\b\d{13,19}\b'),  # Raw card numbers
        ]
        
        # Phone numbers (US format)
        self.phone_patterns = [
            re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
        ]
    
    def _is_whitelisted_ip(self, ip: str) -> bool:
        """Check if IP address should be preserved (common non-sensitive IPs)"""
        whitelist = [
            '127.0.0.1', '0.0.0.0', '255.255.255.255',
            '192.168.1.1', '10.0.0.1', '172.16.0.1',
            '8.8.8.8', '8.8.4.4',  # Google DNS
            '1.1.1.1', '1.0.0.1',  # Cloudflare DNS
        ]
        return ip in whitelist
    
    def _is_example_email(self, email: str) -> bool:
        """Check if email is a common example/placeholder email"""
        email_lower = email.lower()
        example_domains = [
            'example.com', 'example.org', 'test.com', 'sample.com',
            'foo.com', 'bar.com', 'domain.com'
        ]
        example_users = [
            'user@', 'test@', 'example@', 'sample@', 'demo@',
            'foo@', 'bar@', 'admin@localhost'
        ]
        
        return (any(domain in email_lower for domain in example_domains) or 
                any(email_lower.startswith(user) for user in example_users))
    
    def _generate_replacement(self, pii_type: str, original: str) -> str:
        """Generate consistent replacement for PII items"""
        # Create a short hash of the original for consistency
        hash_short = hashlib.md5(original.encode()).hexdigest()[:8]
        
        replacements = {
            'email': f'user@example.com',
            'ip': f'0.0.0.0',
            'api_key': f'[API_KEY_REMOVED]',
            'secret': f'[SECRET_REMOVED]',
            'token': f'[TOKEN_REMOVED]',
            'url': f'https://example.com/path',
            'phone': f'555-000-0000',
            'cc': f'**** **** **** {hash_short[:4]}',
            'db': f'[DB_CONNECTION_REMOVED]'
        }
        
        return replacements.get(pii_type, f'[{pii_type.upper()}_REMOVED]')
    
    def remove_pii_from_text(self, text: str, preserve_structure: bool = True) -> Tuple[str, Dict]:
        """
        Remove PII from text content.
        
        Args:
            text: Input text to clean
            preserve_structure: Whether to preserve code structure when removing PII
            
        Returns:
            Tuple of (cleaned_text, removal_stats)
        """
        cleaned_text = text
        removal_stats = {
            'emails': 0, 'ips': 0, 'api_keys': 0, 
            'secrets': 0, 'urls': 0, 'phones': 0, 'ccs': 0, 'dbs': 0
        }
        
        # Track removed items for logging
        removed_items = []
        
        # Remove emails
        for pattern in self.email_patterns:
            matches = pattern.findall(cleaned_text)
            for match in matches:
                if not self._is_example_email(match):
                    replacement = self._generate_replacement('email', match)
                    cleaned_text = pattern.sub(replacement, cleaned_text, count=1)
                    removal_stats['emails'] += 1
                    removed_items.append(f"Email: {match[:10]}...")
        
        # Remove IP addresses
        for pattern in self.ip_patterns:
            matches = pattern.findall(cleaned_text)
            for match in matches:
                if not self._is_whitelisted_ip(match):
                    replacement = self._generate_replacement('ip', match)
                    cleaned_text = pattern.sub(replacement, cleaned_text, count=1)
                    removal_stats['ips'] += 1
                    removed_items.append(f"IP: {match}")
        
        # Remove API keys and secrets
        for pattern in self.api_key_patterns:
            matches = pattern.findall(cleaned_text)
            for match in matches:
                # match is a tuple (key_name, key_value)
                if len(match) >= 2 and len(match[1]) >= 8:  # Minimum length for real secrets
                    key_name, key_value = match[0], match[1]
                    full_match = f"{key_name}={key_value}"
                    replacement = f"{key_name}=[REDACTED]"
                    cleaned_text = cleaned_text.replace(full_match, replacement)
                    removal_stats['api_keys'] += 1
                    removed_items.append(f"API Key: {key_name}")

        # Remove standalone tokens (not tied to a key assignment)
        for pattern in self.standalone_token_patterns:
            matches = pattern.findall(cleaned_text)
            for token in matches:
                cleaned_text = cleaned_text.replace(token, self._generate_replacement('token', token))
                removal_stats['secrets'] += 1
                removed_items.append("Token: ghp_…")
        
        # Remove URLs (selectively - preserve documentation URLs)
        for pattern in self.url_patterns:
            matches = pattern.findall(cleaned_text)
            for match in matches:
                if not any(safe_domain in match.lower() 
                          for safe_domain in ['github.com', 'stackoverflow.com', 'docs.', 'example.com']):
                    replacement = self._generate_replacement('url', match)
                    cleaned_text = cleaned_text.replace(match, replacement)
                    removal_stats['urls'] += 1
                    removed_items.append(f"URL: {match[:30]}...")
        
        # Remove phone numbers
        for pattern in self.phone_patterns:
            matches = pattern.findall(cleaned_text)
            for match in matches:
                full_match = ''.join(match) if isinstance(match, tuple) else match
                replacement = self._generate_replacement('phone', full_match)
                cleaned_text = pattern.sub(replacement, cleaned_text)
                removal_stats['phones'] += 1
                removed_items.append(f"Phone: {full_match}")
        
        # Log removed items if enabled
        if self.log_removed_items and removed_items:
            self.logger.info(f"PII removed: {', '.join(removed_items[:5])}{'...' if len(removed_items) > 5 else ''}")
        
        return cleaned_text, removal_stats
    
    def process_file(self, input_path: Path, output_path: Path) -> Dict:
        """
        Process a single file to remove PII.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            
        Returns:
            Dictionary with processing statistics
        """
        try:
            # Read input file
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Remove PII
            cleaned_content, removal_stats = self.remove_pii_from_text(content)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write cleaned content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            # Update global statistics
            self.stats['files_processed'] += 1
            for key, count in removal_stats.items():
                if key in self.stats:
                    self.stats[key.rstrip('s') + '_removed'] += count
                self.stats['pii_items_removed'] += count
            
            self.logger.debug(f"Processed {input_path.name}: {sum(removal_stats.values())} PII items removed")
            
            return {
                'success': True,
                'input_file': str(input_path),
                'output_file': str(output_path),
                'pii_removed': sum(removal_stats.values()),
                'details': removal_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {input_path}: {str(e)}")
            return {
                'success': False,
                'input_file': str(input_path),
                'error': str(e)
            }
    
    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset processing statistics"""
        for key in self.stats:
            self.stats[key] = 0

# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    pii_remover = PIIRemover(log_removed_items=True)
    
    # Test with sample text
    sample_code = '''
    # Sample configuration file
    API_KEY = "sk-1234567890abcdef1234567890abcdef"
    EMAIL_USER = "admin@company.com"
    DATABASE_URL = "postgresql://user:pass@192.168.1.100:5432/db"
    GITHUB_TOKEN = "ghp_abcdefghijklmnopqrstuvwxyz123456"
    
    def connect_to_api():
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "User-Agent": "MyApp/1.0 (contact@mycompany.com)"
        }
        return requests.get("https://api.internal.company.com/data", headers=headers)
    
    # Contact: john.doe@company.com for issues
    SERVER_IP = "10.0.1.25"
    '''
    
    cleaned_code, stats = pii_remover.remove_pii_from_text(sample_code)
    
    print("Original code length:", len(sample_code))
    print("Cleaned code length:", len(cleaned_code))
    print("PII removal stats:", stats)
    print("\nCleaned code preview:")
    print(cleaned_code[:500] + "..." if len(cleaned_code) > 500 else cleaned_code)