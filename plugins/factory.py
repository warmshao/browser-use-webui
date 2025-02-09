import os
import yaml
import logging
import importlib.util
import subprocess
from typing import Dict, Any, Optional, Type, Tuple, List
from .plugin_base import PluginBase

logger = logging.getLogger(__name__)

class PluginFactory:
    """Factory for creating and configuring plugins."""
    
    @classmethod
    def _check_version(cls, current: str, required: str) -> bool:
        """Compare version strings."""
        try:
            current_parts = [int(x) for x in current.split('.')]
            required_parts = [int(x) for x in required.split('.')]
            
            # Pad with zeros if lengths don't match
            while len(current_parts) < len(required_parts):
                current_parts.append(0)
            while len(required_parts) < len(current_parts):
                required_parts.append(0)
                
            # Compare version parts
            return current_parts >= required_parts
        except (ValueError, AttributeError):
            logger.error(f"Invalid version format: {current} or {required}")
            return False
    
    @classmethod
    def _get_installed_version(cls, package: str) -> Optional[str]:
        """Get installed package version using pip."""
        try:
            result = subprocess.run(
                ['pip', 'show', package],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Version: '):
                        return line.split('Version: ')[1].strip()
            return None
        except Exception as e:
            logger.error(f"Error checking package version: {str(e)}")
            return None
    
    @classmethod
    def _install_dependencies(cls, dependencies: List[Dict[str, str]]) -> bool:
        """Install plugin dependencies using pip."""
        try:
            if not dependencies:
                return True
                
            logger.info("Checking plugin dependencies...")
            for dep in dependencies:
                package = dep.get('package')
                if not package:
                    continue
                
                # Parse package name and version constraint
                if '>=' in package:
                    pkg_name, version_required = package.split('>=')
                    pkg_name = pkg_name.strip()
                    version_required = version_required.strip()
                else:
                    pkg_name = package.strip()
                    version_required = None
                
                # Check if package is installed with correct version
                current_version = cls._get_installed_version(pkg_name)
                if current_version:
                    if version_required:
                        if cls._check_version(current_version, version_required):
                            logger.info(f"Package {pkg_name} {current_version} already satisfies requirement {package}")
                            continue
                        else:
                            logger.info(f"Updating {pkg_name} {current_version} to {version_required}")
                    else:
                        logger.info(f"Package {pkg_name} {current_version} already installed")
                        continue
                else:
                    logger.info(f"Package {pkg_name} not found, installing...")
                
                # Install or upgrade package
                logger.info(f"Installing {package}")
                result = subprocess.run(
                    ['pip', 'install', package],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.error(f"Failed to install {package}: {result.stderr}")
                    return False
                    
            logger.info("All dependencies are satisfied")
            return True
            
        except Exception as e:
            logger.error(f"Error managing dependencies: {str(e)}")
            return False
    
    @classmethod
    def create_plugin(cls, plugin_dir: str, plugin_name: str) -> Optional[PluginBase]:
        """Create and configure a plugin instance."""
        try:
            # Load manifest first
            manifest = cls._load_manifest(plugin_dir, plugin_name)
            if not manifest:
                return None
            
            # Install dependencies if enabled
            if manifest.get('config', {}).get('auto_install_deps', False):
                if not cls._install_dependencies(manifest.get('dependencies', [])):
                    logger.error(f"Failed to install dependencies for plugin {plugin_name}")
                    return None
            
            # Load plugin class
            plugin_class = cls._load_plugin_class(plugin_dir, plugin_name)
            if not plugin_class:
                return None
            
            # Create plugin instance with manifest
            return cls._instantiate_plugin(plugin_class, manifest)
            
        except Exception as e:
            logger.error(f"Failed to create plugin {plugin_name}: {str(e)}")
            return None
    
    @classmethod
    def _load_manifest(cls, plugin_dir: str, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Load and validate plugin manifest."""
        manifest_path = os.path.join(plugin_dir, plugin_name, "manifest.yaml")
        try:
            with open(manifest_path, 'r') as f:
                manifest = yaml.safe_load(f) or {}
            
            # Validate required fields
            if not manifest.get('name'):
                logger.error(f"Manifest missing required 'name' field: {manifest_path}")
                return None
            if not manifest.get('version'):
                logger.error(f"Manifest missing required 'version' field: {manifest_path}")
                return None
                
            return manifest
            
        except Exception as e:
            logger.error(f"Failed to load manifest from {manifest_path}: {str(e)}")
            return None
    
    @classmethod
    def _load_plugin_class(cls, plugin_dir: str, plugin_name: str) -> Optional[Type[PluginBase]]:
        """Load plugin class from module."""
        plugin_file = os.path.join(plugin_dir, plugin_name, f"{plugin_name}_plugin.py")
        try:
            # Load module
            spec = importlib.util.spec_from_file_location(
                f"plugins.{plugin_name}",
                plugin_file
            )
            if spec is None or spec.loader is None:
                logger.error(f"Failed to load plugin spec: {plugin_name}")
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class
            for item_name in dir(module):
                item = getattr(module, item_name)
                if (isinstance(item, type) and 
                    issubclass(item, PluginBase) and 
                    item != PluginBase and 
                    item_name.endswith('Plugin')):
                    return item
            
            logger.error(f"No valid plugin class found in {plugin_file}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load plugin class from {plugin_file}: {str(e)}")
            return None
    
    @classmethod
    def _instantiate_plugin(cls, plugin_class: Type[PluginBase], manifest: Dict[str, Any]) -> Optional[PluginBase]:
        """Create plugin instance with manifest configuration."""
        try:
            # Create plugin instance with manifest injected
            plugin = plugin_class(manifest=manifest)
            
            # Verify plugin loaded correctly
            if plugin.name != manifest['name']:
                logger.error(
                    f"Plugin name mismatch for {plugin_class.__name__}: "
                    f"manifest name '{manifest['name']}' != plugin name '{plugin.name}'"
                )
                return None
            
            logger.info(f"Successfully created plugin: {plugin.name} v{plugin.version}")
            return plugin
            
        except Exception as e:
            logger.error(f"Failed to instantiate plugin {plugin_class.__name__}: {str(e)}")
            return None 