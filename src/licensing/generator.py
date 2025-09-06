"""License generation functionality for Local RAG Assistant."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
import base64

from ..utils.config import Configuration
from ..utils.logging import get_logger
from ..utils.helpers import ensure_directory


class LicenseGenerator:
    """Generates RSA-based license keys for the RAG assistant."""
    
    def __init__(self, config: Configuration):
        """
        Initialize the license generator.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.licenses_dir = config.paths.licenses
        ensure_directory(self.licenses_dir)
    
    def generate_rsa_keys(self, key_size: Optional[int] = None) -> None:
        """
        Generate RSA key pair for license signing.
        
        Args:
            key_size: RSA key size in bits. If None, uses config value.
        """
        if key_size is None:
            key_size = self.config.licensing.get('key_size', 2048)
        
        try:
            self.logger.info(f"Generating RSA key pair with {key_size} bits")
            
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size
            )
            
            # Get public key
            public_key = private_key.public_key()
            
            # Serialize private key
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            # Serialize public key
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            # Save keys to files
            private_key_path = self.licenses_dir / "private_key.pem"
            public_key_path = self.licenses_dir / "public_key.pem"
            
            with open(private_key_path, 'wb') as f:
                f.write(private_pem)
            
            with open(public_key_path, 'wb') as f:
                f.write(public_pem)
            
            # Set secure permissions (owner read/write only)
            private_key_path.chmod(0o600)
            public_key_path.chmod(0o644)
            
            self.logger.info(f"RSA keys generated and saved to {self.licenses_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate RSA keys: {e}")
            raise
    
    def generate_license(
        self, 
        plan: str = "basic",
        user_id: Optional[str] = None,
        max_queries: Optional[int] = None,
        expiry_days: Optional[int] = None,
        custom_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a signed license token.
        
        Args:
            plan: License plan (e.g., 'basic', 'pro', 'enterprise').
            user_id: User identifier.
            max_queries: Maximum queries allowed.
            expiry_days: Days until license expires.
            custom_data: Additional custom data to include.
            
        Returns:
            Base64-encoded signed license token.
        """
        try:
            # Load private key
            private_key_path = self.licenses_dir / "private_key.pem"
            if not private_key_path.exists():
                raise FileNotFoundError("Private key not found. Generate keys first.")
            
            with open(private_key_path, 'rb') as f:
                private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None
                )
            
            # Create license data
            current_time = int(time.time())
            
            if expiry_days is None:
                expiry_days = self.config.licensing.get('token_expiry_days', 365)
            
            if max_queries is None:
                max_queries = self.config.licensing.get('max_queries_per_day', 1000)
            
            license_data = {
                'plan': plan,
                'user_id': user_id,
                'issued_at': current_time,
                'expires_at': current_time + (expiry_days * 24 * 3600),
                'max_queries_per_day': max_queries,
                'version': '1.0'
            }
            
            # Add custom data if provided
            if custom_data:
                license_data.update(custom_data)
            
            # Serialize license data
            license_json = json.dumps(license_data, sort_keys=True).encode('utf-8')
            
            # Sign the license data
            signature = private_key.sign(
                license_json,
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            
            # Create the complete license token
            license_token = {
                'data': license_data,
                'signature': base64.b64encode(signature).decode('utf-8')
            }
            
            # Encode the complete token
            token_json = json.dumps(license_token).encode('utf-8')
            encoded_token = base64.b64encode(token_json).decode('utf-8')
            
            self.logger.info(f"Generated license for plan '{plan}' expiring in {expiry_days} days")
            
            return encoded_token
            
        except Exception as e:
            self.logger.error(f"Failed to generate license: {e}")
            raise
    
    def save_license(self, token: str, filename: Optional[str] = None) -> Path:
        """
        Save license token to file.
        
        Args:
            token: License token to save.
            filename: Output filename. If None, generates timestamp-based name.
            
        Returns:
            Path to saved license file.
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"license_{timestamp}.txt"
        
        license_path = self.licenses_dir / filename
        
        try:
            with open(license_path, 'w', encoding='utf-8') as f:
                f.write(token)
            
            self.logger.info(f"License saved to {license_path}")
            return license_path
            
        except Exception as e:
            self.logger.error(f"Failed to save license: {e}")
            raise
    
    def get_license_info(self, token: str) -> Dict[str, Any]:
        """
        Extract information from a license token without validating signature.
        
        Args:
            token: License token to inspect.
            
        Returns:
            Dictionary with license information.
        """
        try:
            # Decode the token
            token_json = base64.b64decode(token.encode('utf-8'))
            license_token = json.loads(token_json)
            
            return license_token.get('data', {})
            
        except Exception as e:
            self.logger.error(f"Failed to extract license info: {e}")
            return {}
    
    def create_demo_license(self) -> str:
        """
        Create a demo license with limited functionality.
        
        Returns:
            Demo license token.
        """
        return self.generate_license(
            plan="demo",
            user_id="demo_user",
            max_queries=50,
            expiry_days=7,
            custom_data={
                'features': ['basic_rag', 'limited_models'],
                'restrictions': ['no_commercial_use', 'watermarked_output']
            }
        )
    
    def create_development_license(self) -> str:
        """
        Create a development license for testing.
        
        Returns:
            Development license token.
        """
        return self.generate_license(
            plan="development",
            user_id="dev_user",
            max_queries=10000,
            expiry_days=30,
            custom_data={
                'features': ['full_rag', 'all_models', 'debugging'],
                'environment': 'development'
            }
        )
    
    def keys_exist(self) -> bool:
        """
        Check if RSA key pair exists.
        
        Returns:
            True if both keys exist, False otherwise.
        """
        private_key_path = self.licenses_dir / "private_key.pem"
        public_key_path = self.licenses_dir / "public_key.pem"
        
        return private_key_path.exists() and public_key_path.exists()
    
    def setup_licensing(self) -> Dict[str, str]:
        """
        Complete setup of licensing system.
        
        Returns:
            Dictionary with setup results and sample licenses.
        """
        results = {}
        
        try:
            # Generate keys if they don't exist
            if not self.keys_exist():
                self.generate_rsa_keys()
                results['keys_generated'] = True
            else:
                results['keys_generated'] = False
                self.logger.info("RSA keys already exist")
            
            # Generate sample licenses
            demo_license = self.create_demo_license()
            dev_license = self.create_development_license()
            
            # Save sample licenses
            demo_path = self.save_license(demo_license, "demo_license.txt")
            dev_path = self.save_license(dev_license, "development_license.txt")
            
            results.update({
                'demo_license': demo_license,
                'dev_license': dev_license,
                'demo_license_path': str(demo_path),
                'dev_license_path': str(dev_path),
                'setup_complete': True
            })
            
            self.logger.info("Licensing system setup completed successfully")
            
        except Exception as e:
            results['setup_complete'] = False
            results['error'] = str(e)
            self.logger.error(f"Licensing setup failed: {e}")
        
        return results
