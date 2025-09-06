"""License validation functionality for Local RAG Assistant."""

import json
import time
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
import base64

from ..utils.config import Configuration
from ..utils.logging import get_logger
from ..utils.helpers import ensure_directory


class LicenseValidator:
    """Validates RSA-based license keys for the RAG assistant."""
    
    def __init__(self, config: Configuration):
        """
        Initialize the license validator.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.licenses_dir = config.paths.licenses
        self.usage_db_path = self.licenses_dir / "usage.db"
        
        ensure_directory(self.licenses_dir)
        self._initialize_usage_db()
    
    def _initialize_usage_db(self) -> None:
        """Initialize the usage tracking database."""
        try:
            conn = sqlite3.connect(str(self.usage_db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS license_usage (
                    license_hash TEXT PRIMARY KEY,
                    plan TEXT NOT NULL,
                    user_id TEXT,
                    first_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_queries INTEGER DEFAULT 0,
                    daily_queries INTEGER DEFAULT 0,
                    last_reset_date DATE DEFAULT CURRENT_DATE
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    license_hash TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    query_length INTEGER,
                    response_length INTEGER,
                    processing_time REAL,
                    FOREIGN KEY (license_hash) REFERENCES license_usage(license_hash)
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize usage database: {e}")
            raise
    
    def validate_license(self, token: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a license token.
        
        Args:
            token: License token to validate.
            
        Returns:
            Tuple of (is_valid, validation_info).
        """
        validation_info = {
            'valid': False,
            'reason': '',
            'data': {},
            'signature_valid': False,
            'expired': False,
            'plan': None,
            'remaining_queries': 0
        }
        
        try:
            # Decode the token
            try:
                token_json = base64.b64decode(token.encode('utf-8'))
                license_token = json.loads(token_json)
            except Exception as e:
                validation_info['reason'] = f"Invalid token format: {e}"
                return False, validation_info
            
            # Extract data and signature
            license_data = license_token.get('data', {})
            signature_b64 = license_token.get('signature', '')
            
            if not license_data or not signature_b64:
                validation_info['reason'] = "Missing license data or signature"
                return False, validation_info
            
            validation_info['data'] = license_data
            validation_info['plan'] = license_data.get('plan', 'unknown')
            
            # Verify signature
            signature_valid = self._verify_signature(license_data, signature_b64)
            validation_info['signature_valid'] = signature_valid
            
            if not signature_valid:
                validation_info['reason'] = "Invalid signature"
                return False, validation_info
            
            # Check expiration
            current_time = int(time.time())
            expires_at = license_data.get('expires_at', 0)
            
            if current_time > expires_at:
                validation_info['expired'] = True
                validation_info['reason'] = "License expired"
                return False, validation_info
            
            # Check usage limits
            usage_check = self._check_usage_limits(token, license_data)
            validation_info['remaining_queries'] = usage_check['remaining_queries']
            
            if not usage_check['within_limits']:
                validation_info['reason'] = usage_check['reason']
                return False, validation_info
            
            # All checks passed
            validation_info['valid'] = True
            validation_info['reason'] = "Valid license"
            
            return True, validation_info
            
        except Exception as e:
            validation_info['reason'] = f"Validation error: {e}"
            self.logger.error(f"License validation failed: {e}")
            return False, validation_info
    
    def _verify_signature(self, license_data: Dict[str, Any], signature_b64: str) -> bool:
        """
        Verify the license signature.
        
        Args:
            license_data: License data dictionary.
            signature_b64: Base64-encoded signature.
            
        Returns:
            True if signature is valid, False otherwise.
        """
        try:
            # Load public key
            public_key_path = self.licenses_dir / "public_key.pem"
            if not public_key_path.exists():
                self.logger.error("Public key not found")
                return False
            
            with open(public_key_path, 'rb') as f:
                public_key = serialization.load_pem_public_key(f.read())
            
            # Recreate the signed data
            license_json = json.dumps(license_data, sort_keys=True).encode('utf-8')
            
            # Decode signature
            signature = base64.b64decode(signature_b64.encode('utf-8'))
            
            # Verify signature
            public_key.verify(
                signature,
                license_json,
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Signature verification failed: {e}")
            return False
    
    def _check_usage_limits(self, token: str, license_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if license usage is within limits.
        
        Args:
            token: License token.
            license_data: License data dictionary.
            
        Returns:
            Dictionary with usage check results.
        """
        result = {
            'within_limits': True,
            'reason': '',
            'remaining_queries': 0,
            'daily_queries_used': 0
        }
        
        try:
            # Get license hash for tracking
            license_hash = str(hash(token))
            max_queries_per_day = license_data.get('max_queries_per_day', 1000)
            
            conn = sqlite3.connect(str(self.usage_db_path))
            cursor = conn.cursor()
            
            # Get or create usage record
            cursor.execute("""
                SELECT daily_queries, last_reset_date 
                FROM license_usage 
                WHERE license_hash = ?
            """, (license_hash,))
            
            row = cursor.fetchone()
            current_date = time.strftime('%Y-%m-%d')
            
            if row:
                daily_queries, last_reset_date = row
                
                # Reset daily counter if it's a new day
                if last_reset_date != current_date:
                    cursor.execute("""
                        UPDATE license_usage 
                        SET daily_queries = 0, last_reset_date = ? 
                        WHERE license_hash = ?
                    """, (current_date, license_hash))
                    daily_queries = 0
                    
            else:
                # Create new usage record
                cursor.execute("""
                    INSERT INTO license_usage 
                    (license_hash, plan, user_id, daily_queries, last_reset_date)
                    VALUES (?, ?, ?, 0, ?)
                """, (
                    license_hash,
                    license_data.get('plan', 'unknown'),
                    license_data.get('user_id', 'unknown'),
                    current_date
                ))
                daily_queries = 0
            
            conn.commit()
            conn.close()
            
            # Check limits
            result['daily_queries_used'] = daily_queries
            result['remaining_queries'] = max(0, max_queries_per_day - daily_queries)
            
            if daily_queries >= max_queries_per_day:
                result['within_limits'] = False
                result['reason'] = f"Daily query limit exceeded ({daily_queries}/{max_queries_per_day})"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Usage limit check failed: {e}")
            result['within_limits'] = False
            result['reason'] = f"Usage check error: {e}"
            return result
    
    def record_query_usage(
        self, 
        token: str, 
        query_length: int = 0,
        response_length: int = 0,
        processing_time: float = 0.0
    ) -> bool:
        """
        Record a query usage for a license.
        
        Args:
            token: License token.
            query_length: Length of the query.
            response_length: Length of the response.
            processing_time: Time taken to process the query.
            
        Returns:
            True if recorded successfully, False otherwise.
        """
        try:
            license_hash = str(hash(token))
            
            conn = sqlite3.connect(str(self.usage_db_path))
            cursor = conn.cursor()
            
            # Update usage counters
            cursor.execute("""
                UPDATE license_usage 
                SET total_queries = total_queries + 1,
                    daily_queries = daily_queries + 1,
                    last_used = CURRENT_TIMESTAMP
                WHERE license_hash = ?
            """, (license_hash,))
            
            # Record detailed query log
            cursor.execute("""
                INSERT INTO query_log 
                (license_hash, query_length, response_length, processing_time)
                VALUES (?, ?, ?, ?)
            """, (license_hash, query_length, response_length, processing_time))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record query usage: {e}")
            return False
    
    def get_license_usage(self, token: str) -> Dict[str, Any]:
        """
        Get usage statistics for a license.
        
        Args:
            token: License token.
            
        Returns:
            Dictionary with usage statistics.
        """
        try:
            license_hash = str(hash(token))
            
            conn = sqlite3.connect(str(self.usage_db_path))
            cursor = conn.cursor()
            
            # Get basic usage info
            cursor.execute("""
                SELECT plan, user_id, first_used, last_used, 
                       total_queries, daily_queries, last_reset_date
                FROM license_usage 
                WHERE license_hash = ?
            """, (license_hash,))
            
            row = cursor.fetchone()
            
            if not row:
                conn.close()
                return {'exists': False}
            
            # Get recent query activity
            cursor.execute("""
                SELECT COUNT(*), AVG(processing_time), 
                       MIN(timestamp), MAX(timestamp)
                FROM query_log 
                WHERE license_hash = ? 
                AND timestamp >= datetime('now', '-7 days')
            """, (license_hash,))
            
            activity_row = cursor.fetchone()
            conn.close()
            
            return {
                'exists': True,
                'plan': row[0],
                'user_id': row[1],
                'first_used': row[2],
                'last_used': row[3],
                'total_queries': row[4],
                'daily_queries': row[5],
                'last_reset_date': row[6],
                'recent_activity': {
                    'queries_last_7_days': activity_row[0] if activity_row else 0,
                    'avg_processing_time': activity_row[1] if activity_row else 0,
                    'first_query': activity_row[2] if activity_row else None,
                    'last_query': activity_row[3] if activity_row else None
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get license usage: {e}")
            return {'exists': False, 'error': str(e)}
    
    def load_license_from_file(self, license_path: str) -> Optional[str]:
        """
        Load license token from file.
        
        Args:
            license_path: Path to license file.
            
        Returns:
            License token string or None if failed.
        """
        try:
            with open(license_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
                
        except Exception as e:
            self.logger.error(f"Failed to load license from {license_path}: {e}")
            return None
    
    def is_feature_enabled(self, token: str, feature: str) -> bool:
        """
        Check if a specific feature is enabled for a license.
        
        Args:
            token: License token.
            feature: Feature name to check.
            
        Returns:
            True if feature is enabled, False otherwise.
        """
        is_valid, validation_info = self.validate_license(token)
        
        if not is_valid:
            return False
        
        license_data = validation_info.get('data', {})
        features = license_data.get('features', [])
        
        return feature in features
    
    def get_license_restrictions(self, token: str) -> List[str]:
        """
        Get list of restrictions for a license.
        
        Args:
            token: License token.
            
        Returns:
            List of restriction strings.
        """
        is_valid, validation_info = self.validate_license(token)
        
        if not is_valid:
            return ['invalid_license']
        
        license_data = validation_info.get('data', {})
        return license_data.get('restrictions', [])
    
    def cleanup_old_usage_data(self, days_to_keep: int = 30) -> int:
        """
        Clean up old usage data to save space.
        
        Args:
            days_to_keep: Number of days of data to keep.
            
        Returns:
            Number of records deleted.
        """
        try:
            conn = sqlite3.connect(str(self.usage_db_path))
            cursor = conn.cursor()
            
            # Delete old query logs
            cursor.execute("""
                DELETE FROM query_log 
                WHERE timestamp < datetime('now', '-{} days')
            """.format(days_to_keep))
            
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleaned up {deleted_count} old usage records")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup usage data: {e}")
            return 0
