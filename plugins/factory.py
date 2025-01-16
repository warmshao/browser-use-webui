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
    def _install_dependencies(cls, dependencies: List[Dict[str, str]]) -> bool:
        """Install plugin dependencies using pip."""
        try:
            if not dependencies:
                return True
                
            logger.info("Installing plugin dependencies...")
            for dep in dependencies:
                package = dep.get('package')
                if not package:
                    continue
                    
                logger.info(f"Installing {package}")
                result = subprocess.run(
                    ['pip', 'install', package],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.error(f"Failed to install {package}: {result.stderr}")
                    return False
                    
            logger.info("Successfully installed all dependencies")
            return True
            
        except Exception as e:
            logger.error(f"Error installing dependencies: {str(e)}")
            return False
    
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