# Chronicle Weaver - Phase 4: Image Generation & Advanced Features

**Duration**: 3-4 Weeks  
**Implementation Confidence**: 65% - High Risk  
**Dependencies**: Phase 3 (Agent Management & Core Sub-Agents)  
**Next Phase**: Phase 5 (Polish, Testing & Deployment)

## Overview
Implement advanced features including image generation capabilities, sophisticated sub-agents, and the props/clothing library system. This phase transforms Chronicle Weaver from a text-based assistant into a rich, multimedia roleplaying experience with intelligent agent coordination and visual content generation.

## Key Risk Factors
- **Multiple API integrations** - DALL-E 3, Stability AI, local generation APIs with varying interfaces
- **Local image generation setup complexity** - Automatic1111/ComfyUI configuration and dependencies
- **Advanced agent coordination** - Complex inter-agent communication and conflict resolution
- **API rate limiting and costs** - Managing image generation quotas and expenses
- **File management complexity** - Image storage, thumbnails, metadata handling
- **Performance with large libraries** - Efficient handling of thousands of props and images

## Acceptance Criteria
- [ ] Image generation works with DALL-E 3, Stability AI, and local APIs
- [ ] Generated images display inline in conversation
- [ ] Library tab manages props and clothing items
- [ ] Search Agent retrieves relevant web information
- [ ] Alternate Sub-Ego Agent enables personality switching
- [ ] Prop Agent influences descriptions and image prompts
- [ ] Image generation prompts are context-aware
- [ ] All advanced agents integrate smoothly with main workflow
- [ ] Settings tab configures all external APIs

## Detailed Implementation Steps

### Week 1: Image Generation Foundation

#### 1.1 Image Generation Configuration (`src/image_gen/image_config.py`)

```python
"""Image generation configuration management"""
from dataclasses import dataclass
from typing import Dict, Optional, Any, List
import json
from pathlib import Path
from enum import Enum

class ImageProvider(Enum):
    """Supported image generation providers"""
    DALLE3 = "dalle3"
    STABILITY_AI = "stability_ai"
    AUTOMATIC1111 = "automatic1111"
    COMFYUI = "comfyui"

@dataclass
class ImageGenerationConfig:
    """Configuration for image generation provider"""
    provider: ImageProvider
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    default_size: str = "1024x1024"
    default_quality: str = "standard"
    default_style: str = "natural"
    max_concurrent_requests: int = 3
    timeout: int = 60
    enabled: bool = True
    priority: int = 1  # Lower number = higher priority

@dataclass
class ImagePromptTemplate:
    """Template for generating image prompts"""
    name: str
    template: str
    category: str
    variables: List[str]
    style_modifiers: List[str]
    negative_prompt: Optional[str] = None

class ImageConfigManager:
    """Manages image generation configurations"""
    
    def __init__(self, config_path: str = "config/image_configs.json"):
        self.config_path = Path(config_path)
        self.providers = self._load_provider_configs()
        self.prompt_templates = self._load_prompt_templates()
    
    def _load_provider_configs(self) -> Dict[ImageProvider, ImageGenerationConfig]:
        """Load provider configurations from file"""
        if not self.config_path.exists():
            return self._create_default_provider_configs()
        
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            
            providers = {}
            for provider_name, config_data in data.get("providers", {}).items():
                provider = ImageProvider(provider_name)
                providers[provider] = ImageGenerationConfig(
                    provider=provider,
                    **config_data
                )
            
            return providers
            
        except Exception as e:
            logger.error(f"Error loading image configs: {str(e)}")
            return self._create_default_provider_configs()
    
    def _create_default_provider_configs(self) -> Dict[ImageProvider, ImageGenerationConfig]:
        """Create default provider configurations"""
        defaults = {
            ImageProvider.DALLE3: ImageGenerationConfig(
                provider=ImageProvider.DALLE3,
                model="dall-e-3",
                default_size="1024x1024",
                default_quality="standard",
                priority=1
            ),
            ImageProvider.STABILITY_AI: ImageGenerationConfig(
                provider=ImageProvider.STABILITY_AI,
                base_url="https://api.stability.ai",
                model="stable-diffusion-xl-1024-v1-0",
                default_size="1024x1024",
                priority=2
            ),
            ImageProvider.AUTOMATIC1111: ImageGenerationConfig(
                provider=ImageProvider.AUTOMATIC1111,
                base_url="http://localhost:7860",
                model="sd_xl_base_1.0.safetensors",
                default_size="1024x1024",
                priority=3,
                enabled=False  # Disabled by default
            ),
            ImageProvider.COMFYUI: ImageGenerationConfig(
                provider=ImageProvider.COMFYUI,
                base_url="http://localhost:8188",
                default_size="1024x1024",
                priority=4,
                enabled=False  # Disabled by default
            )
        }
        
        self._save_provider_configs(defaults)
        return defaults
    
    def _load_prompt_templates(self) -> Dict[str, ImagePromptTemplate]:
        """Load image prompt templates"""
        templates_path = self.config_path.parent / "image_prompt_templates.json"
        
        if not templates_path.exists():
            return self._create_default_prompt_templates()
        
        try:
            with open(templates_path, 'r') as f:
                data = json.load(f)
            
            templates = {}
            for template_data in data.get("templates", []):
                template = ImagePromptTemplate(**template_data)
                templates[template.name] = template
            
            return templates
            
        except Exception as e:
            logger.error(f"Error loading prompt templates: {str(e)}")
            return self._create_default_prompt_templates()
    
    def _create_default_prompt_templates(self) -> Dict[str, ImagePromptTemplate]:
        """Create default prompt templates"""
        templates = {
            "character_portrait": ImagePromptTemplate(
                name="character_portrait",
                template="A detailed portrait of {character_name}, {description}, {style_modifiers}",
                category="character",
                variables=["character_name", "description"],
                style_modifiers=["high quality", "detailed", "professional artwork"],
                negative_prompt="blurry, low quality, distorted"
            ),
            "scene_setting": ImagePromptTemplate(
                name="scene_setting",
                template="A {scene_type} showing {location}, {atmosphere}, {style_modifiers}",
                category="scene",
                variables=["scene_type", "location", "atmosphere"],
                style_modifiers=["cinematic", "detailed", "atmospheric"],
                negative_prompt="blurry, low quality"
            ),
            "item_prop": ImagePromptTemplate(
                name="item_prop",
                template="A detailed image of {item_name}, {description}, {style_modifiers}",
                category="prop",
                variables=["item_name", "description"],
                style_modifiers=["isolated on white background", "high detail", "professional product photo"],
                negative_prompt="cluttered background, blurry"
            )
        }
        
        # Save default templates
        templates_path = self.config_path.parent / "image_prompt_templates.json"
        templates_path.parent.mkdir(parents=True, exist_ok=True)
        
        serializable = {
            "templates": [
                {
                    "name": template.name,
                    "template": template.template,
                    "category": template.category,
                    "variables": template.variables,
                    "style_modifiers": template.style_modifiers,
                    "negative_prompt": template.negative_prompt
                }
                for template in templates.values()
            ]
        }
        
        with open(templates_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        return templates
    
    def get_enabled_providers(self) -> List[ImageGenerationConfig]:
        """Get list of enabled providers sorted by priority"""
        enabled = [config for config in self.providers.values() if config.enabled]
        return sorted(enabled, key=lambda x: x.priority)
    
    def get_provider_config(self, provider: ImageProvider) -> Optional[ImageGenerationConfig]:
        """Get configuration for specific provider"""
        return self.providers.get(provider)
    
    def update_provider_config(self, provider: ImageProvider, config: ImageGenerationConfig):
        """Update provider configuration"""
        self.providers[provider] = config
        self._save_provider_configs(self.providers)
    
    def _save_provider_configs(self, providers: Dict[ImageProvider, ImageGenerationConfig]):
        """Save provider configurations to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        serializable = {
            "providers": {
                provider.value: {
                    "api_key": config.api_key,
                    "base_url": config.base_url,
                    "model": config.model,
                    "default_size": config.default_size,
                    "default_quality": config.default_quality,
                    "default_style": config.default_style,
                    "max_concurrent_requests": config.max_concurrent_requests,
                    "timeout": config.timeout,
                    "enabled": config.enabled,
                    "priority": config.priority
                }
                for provider, config in providers.items()
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(serializable, f, indent=2)
```

#### 1.2 Image Manager Core (`src/image_gen/image_manager.py`)

```python
"""Central image generation management"""
import asyncio
import logging
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from datetime import datetime
import hashlib
import json
from .image_config import ImageConfigManager, ImageProvider, ImageGenerationConfig
from .providers import (
    DalleProvider, StabilityProvider, 
    Automatic1111Provider, ComfyUIProvider
)
from ..memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class ImageGenerationRequest:
    """Represents an image generation request"""
    
    def __init__(self, 
                 prompt: str,
                 negative_prompt: Optional[str] = None,
                 size: str = "1024x1024",
                 quality: str = "standard",
                 style: str = "natural",
                 provider_preference: Optional[ImageProvider] = None,
                 metadata: Dict[str, Any] = None):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.size = size
        self.quality = quality
        self.style = style
        self.provider_preference = provider_preference
        self.metadata = metadata or {}
        self.request_id = self._generate_request_id()
        self.created_at = datetime.utcnow()
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        content = f"{self.prompt}_{self.size}_{datetime.utcnow().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

class ImageGenerationResult:
    """Represents the result of image generation"""
    
    def __init__(self,
                 success: bool,
                 image_path: Optional[Path] = None,
                 provider_used: Optional[ImageProvider] = None,
                 generation_time: Optional[float] = None,
                 error_message: Optional[str] = None,
                 metadata: Dict[str, Any] = None):
        self.success = success
        self.image_path = image_path
        self.provider_used = provider_used
        self.generation_time = generation_time
        self.error_message = error_message
        self.metadata = metadata or {}

class ImageManager:
    """Central manager for all image generation operations"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.config_manager = ImageConfigManager()
        self.memory_manager = memory_manager
        self.providers = self._initialize_providers()
        self.generation_queue = asyncio.Queue()
        self.active_generations = {}
        
        # Image storage setup
        self.images_dir = Path("data/images")
        self.thumbnails_dir = Path("data/thumbnails")
        self._setup_storage_directories()
        
        logger.info("Image Manager initialized")
    
    def _initialize_providers(self) -> Dict[ImageProvider, Any]:
        """Initialize all image generation providers"""
        providers = {}
        
        # Initialize each provider
        providers[ImageProvider.DALLE3] = DalleProvider(
            self.config_manager.get_provider_config(ImageProvider.DALLE3)
        )
        providers[ImageProvider.STABILITY_AI] = StabilityProvider(
            self.config_manager.get_provider_config(ImageProvider.STABILITY_AI)
        )
        providers[ImageProvider.AUTOMATIC1111] = Automatic1111Provider(
            self.config_manager.get_provider_config(ImageProvider.AUTOMATIC1111)
        )
        providers[ImageProvider.COMFYUI] = ComfyUIProvider(
            self.config_manager.get_provider_config(ImageProvider.COMFYUI)
        )
        
        return providers
    
    def _setup_storage_directories(self):
        """Setup image storage directories"""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories by date
        today = datetime.now().strftime("%Y/%m")
        (self.images_dir / today).mkdir(parents=True, exist_ok=True)
        (self.thumbnails_dir / today).mkdir(parents=True, exist_ok=True)
    
    async def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """Generate image using best available provider"""
        start_time = datetime.utcnow()
        
        try:
            # Get list of providers to try
            providers_to_try = self._get_providers_for_request(request)
            
            if not providers_to_try:
                return ImageGenerationResult(
                    success=False,
                    error_message="No available image generation providers"
                )
            
            # Try providers in order of preference
            last_error = None
            for provider_config in providers_to_try:
                try:
                    provider = self.providers[provider_config.provider]
                    
                    if not await provider.is_available():
                        logger.warning(f"Provider {provider_config.provider.value} not available")
                        continue
                    
                    # Generate image
                    logger.info(f"Generating image with {provider_config.provider.value}")
                    result_data = await provider.generate_image(
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt,
                        size=request.size,
                        quality=request.quality,
                        style=request.style
                    )
                    
                    if result_data.get("success"):
                        # Save image and create result
                        image_path = await self._save_generated_image(
                            result_data["image_data"],
                            request,
                            provider_config.provider
                        )
                        
                        generation_time = (datetime.utcnow() - start_time).total_seconds()
                        
                        # Log successful generation
                        await self._log_generation_success(request, provider_config.provider, generation_time)
                        
                        return ImageGenerationResult(
                            success=True,
                            image_path=image_path,
                            provider_used=provider_config.provider,
                            generation_time=generation_time,
                            metadata=result_data.get("metadata", {})
                        )
                    else:
                        last_error = result_data.get("error", "Unknown provider error")
                        
                except Exception as e:
                    last_error = str(e)
                    logger.error(f"Error with provider {provider_config.provider.value}: {str(e)}")
                    continue
            
            # All providers failed
            return ImageGenerationResult(
                success=False,
                error_message=f"All providers failed. Last error: {last_error}"
            )
            
        except Exception as e:
            logger.error(f"Error in image generation: {str(e)}")
            return ImageGenerationResult(
                success=False,
                error_message=str(e)
            )
    
    def _get_providers_for_request(self, request: ImageGenerationRequest) -> List[ImageGenerationConfig]:
        """Get ordered list of providers to try for request"""
        enabled_providers = self.config_manager.get_enabled_providers()
        
        if request.provider_preference:
            # Move preferred provider to front
            preferred_config = self.config_manager.get_provider_config(request.provider_preference)
            if preferred_config and preferred_config.enabled:
                # Remove from list and add to front
                enabled_providers = [p for p in enabled_providers if p.provider != request.provider_preference]
                enabled_providers.insert(0, preferred_config)
        
        return enabled_providers
    
    async def _save_generated_image(self, 
                                  image_data: bytes, 
                                  request: ImageGenerationRequest,
                                  provider: ImageProvider) -> Path:
        """Save generated image to disk"""
        # Create filename with timestamp and request ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{request.request_id}.png"
        
        # Create path with date subdirectory
        date_subdir = datetime.now().strftime("%Y/%m")
        image_path = self.images_dir / date_subdir / filename
        
        # Ensure directory exists
        image_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save image
        with open(image_path, 'wb') as f:
            f.write(image_data)
        
        # Generate thumbnail
        await self._generate_thumbnail(image_path)
        
        # Save metadata
        await self._save_image_metadata(image_path, request, provider)
        
        logger.info(f"Saved generated image: {image_path}")
        return image_path
    
    async def _generate_thumbnail(self, image_path: Path):
        """Generate thumbnail for image"""
        try:
            from PIL import Image
            
            # Open and resize image
            with Image.open(image_path) as img:
                img.thumbnail((256, 256), Image.Resampling.LANCZOS)
                
                # Save thumbnail
                thumbnail_path = self.thumbnails_dir / image_path.relative_to(self.images_dir)
                thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
                
                img.save(thumbnail_path, "PNG")
                
        except Exception as e:
            logger.warning(f"Failed to generate thumbnail for {image_path}: {str(e)}")
    
    async def _save_image_metadata(self, 
                                 image_path: Path, 
                                 request: ImageGenerationRequest,
                                 provider: ImageProvider):
        """Save image metadata to JSON file"""
        metadata_path = image_path.with_suffix('.json')
        
        metadata = {
            "request_id": request.request_id,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "size": request.size,
            "quality": request.quality,
            "style": request.style,
            "provider": provider.value,
            "created_at": request.created_at.isoformat(),
            "image_path": str(image_path),
            "metadata": request.metadata
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    async def _log_generation_success(self, 
                                    request: ImageGenerationRequest,
                                    provider: ImageProvider,
                                    generation_time: float):
        """Log successful image generation for analytics"""
        try:
            # This could be expanded to save to database for analytics
            logger.info(f"Image generated successfully: {request.request_id} "
                       f"using {provider.value} in {generation_time:.2f}s")
        except Exception as e:
            logger.warning(f"Failed to log generation success: {str(e)}")
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all providers"""
        status = {}
        
        for provider_type, provider in self.providers.items():
            config = self.config_manager.get_provider_config(provider_type)
            status[provider_type.value] = {
                "enabled": config.enabled if config else False,
                "available": False,  # Will be checked asynchronously
                "priority": config.priority if config else 999,
                "model": config.model if config else None
            }
        
        return status
    
    async def test_provider(self, provider: ImageProvider) -> bool:
        """Test if a specific provider is working"""
        try:
            provider_instance = self.providers.get(provider)
            if not provider_instance:
                return False
            
            return await provider_instance.is_available()
            
        except Exception as e:
            logger.error(f"Error testing provider {provider.value}: {str(e)}")
            return False
    
    def get_recent_images(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recently generated images"""
        images = []
        
        try:
            # Scan image directory for recent files
            for image_path in sorted(self.images_dir.rglob("*.png"), 
                                   key=lambda p: p.stat().st_mtime, 
                                   reverse=True)[:limit]:
                
                metadata_path = image_path.with_suffix('.json')
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check if thumbnail exists
                    thumbnail_path = self.thumbnails_dir / image_path.relative_to(self.images_dir)
                    
                    images.append({
                        "image_path": str(image_path),
                        "thumbnail_path": str(thumbnail_path) if thumbnail_path.exists() else None,
                        "prompt": metadata.get("prompt", ""),
                        "provider": metadata.get("provider", "unknown"),
                        "created_at": metadata.get("created_at", ""),
                        "size": metadata.get("size", "unknown")
                    })
            
            return images
            
        except Exception as e:
            logger.error(f"Error getting recent images: {str(e)}")
            return []
```

### Week 2: Provider Implementations

#### 2.1 DALL-E 3 Provider (`src/image_gen/providers/dalle_provider.py`)

```python
"""DALL-E 3 image generation provider"""
import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional
from ..image_config import ImageGenerationConfig, ImageProvider

logger = logging.getLogger(__name__)

class DalleProvider:
    """DALL-E 3 image generation provider"""
    
    def __init__(self, config: ImageGenerationConfig):
        self.config = config
        self.api_base = "https://api.openai.com/v1"
    
    async def is_available(self) -> bool:
        """Check if DALL-E 3 is available"""
        if not self.config or not self.config.enabled:
            return False
        
        if not self.config.api_key:
            logger.warning("DALL-E 3 API key not configured")
            return False
        
        try:
            # Test API connectivity
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_base}/models",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"DALL-E 3 availability check failed: {str(e)}")
            return False
    
    async def generate_image(self,
                           prompt: str,
                           negative_prompt: Optional[str] = None,
                           size: str = "1024x1024",
                           quality: str = "standard",
                           style: str = "natural") -> Dict[str, Any]:
        """Generate image using DALL-E 3"""
        try:
            # Prepare request data
            data = {
                "model": self.config.model or "dall-e-3",
                "prompt": prompt,
                "size": size,
                "quality": quality,
                "style": style,
                "n": 1,
                "response_format": "b64_json"
            }
            
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_base}/images/generations",
                    json=data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"DALL-E 3 API error: {response.status} - {error_text}"
                        }
                    
                    result = await response.json()
                    
                    if "data" not in result or not result["data"]:
                        return {
                            "success": False,
                            "error": "No image data returned from DALL-E 3"
                        }
                    
                    # Extract base64 image data
                    import base64
                    image_b64 = result["data"][0]["b64_json"]
                    image_data = base64.b64decode(image_b64)
                    
                    return {
                        "success": True,
                        "image_data": image_data,
                        "metadata": {
                            "revised_prompt": result["data"][0].get("revised_prompt"),
                            "model": data["model"],
                            "size": size,
                            "quality": quality,
                            "style": style
                        }
                    }
                    
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "DALL-E 3 request timed out"
            }
        except Exception as e:
            logger.error(f"DALL-E 3 generation error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
```

#### 2.2 Stability AI Provider (`src/image_gen/providers/stability_provider.py`)

```python
"""Stability AI image generation provider"""
import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional
from ..image_config import ImageGenerationConfig, ImageProvider

logger = logging.getLogger(__name__)

class StabilityProvider:
    """Stability AI image generation provider"""
    
    def __init__(self, config: ImageGenerationConfig):
        self.config = config
        self.api_base = config.base_url or "https://api.stability.ai"
    
    async def is_available(self) -> bool:
        """Check if Stability AI is available"""
        if not self.config or not self.config.enabled:
            return False
        
        if not self.config.api_key:
            logger.warning("Stability AI API key not configured")
            return False
        
        try:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Accept": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_base}/v1/user/account",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.error(f"Stability AI availability check failed: {str(e)}")
            return False
    
    async def generate_image(self,
                           prompt: str,
                           negative_prompt: Optional[str] = None,
                           size: str = "1024x1024",
                           quality: str = "standard",
                           style: str = "natural") -> Dict[str, Any]:
        """Generate image using Stability AI"""
        try:
            # Parse size
            width, height = map(int, size.split('x'))
            
            # Prepare request data
            data = {
                "text_prompts": [
                    {
                        "text": prompt,
                        "weight": 1.0
                    }
                ],
                "cfg_scale": 7,
                "height": height,
                "width": width,
                "samples": 1,
                "steps": 30
            }
            
            if negative_prompt:
                data["text_prompts"].append({
                    "text": negative_prompt,
                    "weight": -1.0
                })
            
            # Map quality to steps
            if quality == "hd":
                data["steps"] = 50
            
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            model = self.config.model or "stable-diffusion-xl-1024-v1-0"
            endpoint = f"{self.api_base}/v1/generation/{model}/text-to-image"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"Stability AI error: {response.status} - {error_text}"
                        }
                    
                    result = await response.json()
                    
                    if "artifacts" not in result or not result["artifacts"]:
                        return {
                            "success": False,
                            "error": "No image data returned from Stability AI"
                        }
                    
                    # Extract base64 image data
                    import base64
                    image_b64 = result["artifacts"][0]["base64"]
                    image_data = base64.b64decode(image_b64)
                    
                    return {
                        "success": True,
                        "image_data": image_data,
                        "metadata": {
                            "model": model,
                            "size": size,
                            "steps": data["steps"],
                            "cfg_scale": data["cfg_scale"],
                            "seed": result["artifacts"][0].get("seed")
                        }
                    }
                    
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Stability AI request timed out"
            }
        except Exception as e:
            logger.error(f"Stability AI generation error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
```

### Week 3: Advanced Agents Implementation

#### 3.1 Search Agent (`src/agents/search_agent.py`)

```python
"""Search Agent for web information retrieval"""
import asyncio
import aiohttp
import logging
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentMessage, AgentCapability
from ..llm.llm_manager import LLMManager

logger = logging.getLogger(__name__)

class SearchAgent(BaseAgent):
    """Agent that performs web searches and provides relevant information"""
    
    def __init__(self, llm_manager: LLMManager, search_api_key: str = None):
        super().__init__(
            name="Search Agent",
            system_prompt=self._get_system_prompt(),
            capabilities=[
                AgentCapability(
                    name="web_search",
                    description="Search the web for information"
                ),
                AgentCapability(
                    name="fact_checking",
                    description="Verify facts and provide sources"
                ),
                AgentCapability(
                    name="research_assistance",
                    description="Assist with research and information gathering"
                )
            ]
        )
        self.llm_manager = llm_manager
        self.search_api_key = search_api_key
        self.search_api_base = "https://serpapi.com/search"
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for search agent"""
        return """You are a Search Agent specialized in finding and summarizing relevant information from web searches.

Your responsibilities:
1. Analyze user queries to determine what information is needed
2. Perform targeted web searches using appropriate keywords
3. Summarize search results in a clear and concise manner
4. Provide source links for verification
5. Distinguish between facts and opinions in search results
6. Identify potential biases or conflicting information

Guidelines:
- Always provide sources for information
- Summarize complex information clearly
- Highlight the most relevant and recent information
- Note when information is disputed or uncertain
- Respect privacy and avoid sharing personal information found in searches
- Focus on authoritative and reliable sources

Format your responses with clear headings and bullet points when appropriate."""
    
    async def process_message(self, message: str, context: Dict[str, Any] = None) -> AgentMessage:
        """Process search request and return results"""
        try:
            # Extract search intent from message
            search_query = await self._extract_search_query(message, context)
            
            if not search_query:
                return AgentMessage(
                    content="I couldn't determine what to search for. Could you please clarify what information you're looking for?",
                    metadata={"agent": self.name, "error": "no_search_query"}
                )
            
            # Perform web search
            search_results = await self._perform_web_search(search_query)
            
            if not search_results:
                return AgentMessage(
                    content=f"I couldn't find any relevant information for '{search_query}'. Would you like me to try a different search approach?",
                    metadata={"agent": self.name, "search_query": search_query}
                )
            
            # Analyze and summarize results
            summary = await self._summarize_search_results(search_results, message, context)
            
            # Add to conversation history
            response = AgentMessage(
                content=summary,
                metadata={
                    "agent": self.name,
                    "search_query": search_query,
                    "results_count": len(search_results),
                    "sources": [result.get("link") for result in search_results[:5]]
                }
            )
            
            self.add_to_history(response)
            return response
            
        except Exception as e:
            logger.error(f"Search Agent error: {str(e)}")
            return AgentMessage(
                content="I encountered an error while searching for information. Please try again.",
                metadata={"agent": self.name, "error": str(e)}
            )
    
    async def _extract_search_query(self, message: str, context: Dict[str, Any] = None) -> Optional[str]:
        """Extract search query from user message"""
        try:
            # Use LLM to extract search intent
            messages = [
                {
                    "role": "system",
                    "content": """Extract the main search query from the user's message. 
                    Return only the search terms that would be effective for a web search.
                    If no clear search intent is found, return 'NONE'.
                    
                    Examples:
                    User: "What is the weather like in Tokyo today?" -> "Tokyo weather today"
                    User: "Tell me about the history of jazz music" -> "history of jazz music"
                    User: "Hello" -> "NONE"
                    """
                },
                {
                    "role": "user",
                    "content": message
                }
            ]
            
            response_content = ""
            async for chunk in self.llm_manager.generate_response(messages, stream=False):
                response_content += chunk
            
            search_query = response_content.strip()
            
            if search_query.upper() == "NONE":
                return None
            
            return search_query
            
        except Exception as e:
            logger.error(f"Error extracting search query: {str(e)}")
            # Fallback: use the original message as search query
            return message[:100]  # Truncate to reasonable length
    
    async def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search using SerpAPI"""
        if not self.search_api_key:
            logger.warning("Search API key not configured")
            return []
        
        try:
            params = {
                "q": query,
                "api_key": self.search_api_key,
                "engine": "google",
                "num": 10,
                "hl": "en",
                "gl": "us"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.search_api_base,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status != 200:
                        logger.error(f"Search API error: {response.status}")
                        return []
                    
                    data = await response.json()
                    return data.get("organic_results", [])
                    
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return []
    
    async def _summarize_search_results(self, 
                                      results: List[Dict[str, Any]], 
                                      original_query: str,
                                      context: Dict[str, Any] = None) -> str:
        """Summarize search results using LLM"""
        try:
            # Prepare search results for summarization
            results_text = ""
            for i, result in enumerate(results[:5], 1):
                title = result.get("title", "No title")
                snippet = result.get("snippet", "No description")
                link = result.get("link", "No link")
                
                results_text += f"{i}. {title}\n{snippet}\nSource: {link}\n\n"
            
            # Create summarization prompt
            messages = [
                {
                    "role": "system",
                    "content": """You are a Search Agent summarizing web search results. 
                    
                    Your task:
                    1. Analyze the search results and extract the most relevant information
                    2. Create a clear, well-organized summary
                    3. Include specific facts, figures, and key points
                    4. Always cite sources using [Source 1], [Source 2], etc.
                    5. Note any conflicting information or uncertainties
                    6. Structure your response with headings and bullet points when appropriate
                    
                    Format:
                    ## Summary
                    [Main findings]
                    
                    ## Key Points
                    - Point 1 [Source X]
                    - Point 2 [Source Y]
                    
                    ## Sources
                    1. [First source title and link]
                    2. [Second source title and link]
                    """
                },
                {
                    "role": "user",
                    "content": f"Original query: {original_query}\n\nSearch results:\n{results_text}\n\nPlease summarize these results."
                }
            ]
            
            response_content = ""
            async for chunk in self.llm_manager.generate_response(messages, stream=False):
                response_content += chunk
            
            return response_content.strip()
            
        except Exception as e:
            logger.error(f"Error summarizing search results: {str(e)}")
            
            # Fallback: basic summary
            summary = f"I found {len(results)} results for your search. Here are the top findings:\n\n"
            for i, result in enumerate(results[:3], 1):
                title = result.get("title", "No title")
                snippet = result.get("snippet", "No description")
                link = result.get("link", "")
                summary += f"{i}. **{title}**\n{snippet}\n[Source]({link})\n\n"
            
            return summary
    
    def is_search_relevant(self, message: str) -> bool:
        """Check if message contains search-worthy content"""
        search_indicators = [
            "what is", "who is", "when did", "where is", "how does",
            "tell me about", "information about", "facts about",
            "research", "find out", "look up", "search for"
        ]
        
        message_lower = message.lower()
        return any(indicator in message_lower for indicator in search_indicators)
```

#### 3.2 Props Agent (`src/agents/prop_agent.py`)

```python
"""Props Agent for managing and suggesting props/clothing items"""
import logging
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentMessage, AgentCapability
from ..llm.llm_manager import LLMManager
from ..memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class PropAgent(BaseAgent):
    """Agent that manages props and clothing items"""
    
    def __init__(self, llm_manager: LLMManager, memory_manager: MemoryManager):
        super().__init__(
            name="Prop Agent",
            system_prompt=self._get_system_prompt(),
            capabilities=[
                AgentCapability(
                    name="prop_suggestions",
                    description="Suggest relevant props for scenes"
                ),
                AgentCapability(
                    name="item_description",
                    description="Enhance descriptions with prop details"
                ),
                AgentCapability(
                    name="inventory_management",
                    description="Manage character inventories"
                ),
                AgentCapability(
                    name="image_prompt_enhancement",
                    description="Enhance image generation prompts with prop details"
                )
            ]
        )
        self.llm_manager = llm_manager
        self.memory_manager = memory_manager
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for prop agent"""
        return """You are a Prop Agent specialized in managing and suggesting props, clothing, and items for roleplaying scenarios.

Your responsibilities:
1. Suggest appropriate props and items based on scene context
2. Maintain character inventories and track item usage
3. Enhance descriptions by incorporating relevant item details
4. Recommend items that fit the setting, time period, and character
5. Help with costume and prop selection for different scenarios
6. Enhance image generation prompts with detailed item descriptions

Guidelines:
- Consider the setting, time period, and genre when suggesting items
- Match props to character personality and background
- Provide detailed descriptions that enhance immersion
- Track item rarity and availability
- Suggest alternatives when specific items aren't available
- Consider practical aspects (weight, size, functionality)
- Maintain consistency with established character inventories

When suggesting props:
- Explain why the item fits the scenario
- Provide detailed visual descriptions
- Note any special properties or significance
- Consider how the item might be used in roleplay"""
    
    async def process_message(self, message: str, context: Dict[str, Any] = None) -> AgentMessage:
        """Process prop-related requests"""
        try:
            # Determine the type of prop request
            request_type = await self._classify_prop_request(message, context)
            
            if request_type == "suggestion":
                response_content = await self._suggest_props(message, context)
            elif request_type == "description":
                response_content = await self._enhance_description(message, context)
            elif request_type == "inventory":
                response_content = await self._manage_inventory(message, context)
            elif request_type == "image_enhancement":
                response_content = await self._enhance_image_prompt(message, context)
            else:
                response_content = await self._general_prop_assistance(message, context)
            
            response = AgentMessage(
                content=response_content,
                metadata={
                    "agent": self.name,
                    "request_type": request_type,
                    "props_suggested": self._extract_prop_names(response_content)
                }
            )
            
            self.add_to_history(response)
            return response
            
        except Exception as e:
            logger.error(f"Prop Agent error: {str(e)}")
            return AgentMessage(
                content="I encountered an error while processing your prop request. Please try again.",
                metadata={"agent": self.name, "error": str(e)}
            )
    
    async def _classify_prop_request(self, message: str, context: Dict[str, Any] = None) -> str:
        """Classify the type of prop request"""
        try:
            classification_prompt = f"""
            Classify this message into one of these categories:
            - suggestion: User wants prop/item suggestions for a scenario
            - description: User wants enhanced descriptions of items/scenes
            - inventory: User wants to manage character inventory
            - image_enhancement: User wants to enhance image generation prompts
            - general: General prop-related assistance
            
            Message: {message}
            
            Respond with only the category name.
            """
            
            messages = [
                {"role": "system", "content": "You are a classifier for prop-related requests."},
                {"role": "user", "content": classification_prompt}
            ]
            
            response_content = ""
            async for chunk in self.llm_manager.generate_response(messages, stream=False):
                response_content += chunk
            
            classification = response_content.strip().lower()
            
            valid_types = ["suggestion", "description", "inventory", "image_enhancement", "general"]
            return classification if classification in valid_types else "general"
            
        except Exception as e:
            logger.error(f"Error classifying prop request: {str(e)}")
            return "general"
    
    async def _suggest_props(self, message: str, context: Dict[str, Any] = None) -> str:
        """Suggest appropriate props for the scenario"""
        try:
            # Get relevant props from library
            relevant_props = await self._get_relevant_props(message, context)
            
            # Build suggestion prompt
            props_context = ""
            if relevant_props:
                props_context = "Available props from library:\n"
                for prop in relevant_props:
                    props_context += f"- {prop['name']}: {prop['description']}\n"
            
            suggestion_prompt = f"""
            Based on the user's request, suggest appropriate props and items.
            
            User request: {message}
            {props_context}
            
            Consider:
            - Setting and time period
            - Character background and personality
            - Scene requirements
            - Item functionality and significance
            
            Provide 3-5 specific suggestions with:
            1. Item name
            2. Detailed description
            3. Why it fits the scenario
            4. How it might be used
            """
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": suggestion_prompt}
            ]
            
            response_content = ""
            async for chunk in self.llm_manager.generate_response(messages, stream=True):
                response_content += chunk
            
            return response_content.strip()
            
        except Exception as e:
            logger.error(f"Error suggesting props: {str(e)}")
            return "I encountered an error while suggesting props. Please try again."
    
    async def _get_relevant_props(self, message: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get relevant props from the library based on context"""
        try:
            # Extract keywords for searching
            keywords = await self._extract_prop_keywords(message)
            
            # Search props library
            relevant_props = []
            for keyword in keywords:
                props = self.memory_manager.get_prop_items(search_term=keyword, limit=5)
                relevant_props.extend(props)
            
            # Remove duplicates and limit results
            seen_ids = set()
            unique_props = []
            for prop in relevant_props:
                if prop['id'] not in seen_ids:
                    seen_ids.add(prop['id'])
                    unique_props.append(prop)
                    if len(unique_props) >= 10:
                        break
            
            return unique_props
            
        except Exception as e:
            logger.error(f"Error getting relevant props: {str(e)}")
            return []
    
    async def _extract_prop_keywords(self, message: str) -> List[str]:
        """Extract keywords for prop searching"""
        try:
            keyword_prompt = f"""
            Extract relevant keywords for searching props/items from this message.
            Focus on: objects, clothing, weapons, tools, accessories, settings, themes.
            
            Message: {message}
            
            Return 3-5 keywords separated by commas.
            """
            
            messages = [
                {"role": "system", "content": "You extract keywords for prop searching."},
                {"role": "user", "content": keyword_prompt}
            ]
            
            response_content = ""
            async for chunk in self.llm_manager.generate_response(messages, stream=False):
                response_content += chunk
            
            keywords = [kw.strip() for kw in response_content.split(',')]
            return keywords[:5]  # Limit to 5 keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def _extract_prop_names(self, content: str) -> List[str]:
        """Extract prop names mentioned in response"""
        # Simple extraction - could be enhanced with NLP
        prop_names = []
        lines = content.split('\n')
        for line in lines:
            if any(marker in line for marker in ['1.', '2.', '3.', '4.', '5.', '-', '*']):
                # Extract item name (first few words after marker)
                words = line.split()
                if len(words) > 1:
                    # Find the item name (usually after marker and before colon)
                    name_parts = []
                    start_found = False
                    for word in words:
                        if word in ['1.', '2.', '3.', '4.', '5.', '-', '*']:
                            start_found = True
                            continue
                        if start_found:
                            if ':' in word:
                                name_parts.append(word.replace(':', ''))
                                break
                            name_parts.append(word)
                            if len(name_parts) >= 3:  # Limit name length
                                break
                    
                    if name_parts:
                        prop_names.append(' '.join(name_parts))
        
        return prop_names[:5]  # Limit to 5 extracted names
```

## Testing Strategy

### Unit Tests (`tests/unit/image_gen/`, `tests/unit/agents/`)
- **Image Provider Tests**: Test each provider independently with mock responses
- **Image Manager Tests**: Test provider selection, fallback logic, file operations
- **Agent Tests**: Test message processing, capability management
- **Configuration Tests**: Test config loading, validation, and updates

### Integration Tests (`tests/integration/`)
- **Image Generation Flow**: Test complete image generation pipeline
- **Agent Coordination**: Test multi-agent workflows
- **API Integration**: Test external API connections with error handling
- **File Management**: Test image storage, thumbnails, metadata

### Performance Tests
- **Image Generation Speed**: Benchmark generation times across providers
- **Library Performance**: Test prop library with large datasets
- **Concurrent Operations**: Test multiple simultaneous image generations
- **Memory Usage**: Monitor memory consumption during operations

## Error Handling Strategy
- **API Failures**: Provider fallback system with graceful degradation
- **Network Issues**: Retry logic with exponential backoff
- **File System Errors**: Robust file handling with cleanup
- **Agent Conflicts**: Priority-based conflict resolution

## Success Metrics
- [ ] All image providers generate images successfully
- [ ] Library management handles 1000+ items smoothly
- [ ] Advanced agents provide relevant, helpful responses
- [ ] Image generation completes within 30 seconds
- [ ] Agent coordination works without conflicts
- [ ] Settings UI allows complete configuration

## Deliverables
1. **Image Generation System** - Multi-provider support with fallback
2. **Props Library Management** - Complete CRUD operations with search
3. **Advanced Agent Suite** - Search, Prop, and Alternate Sub-Ego agents
4. **Library UI Tab** - Visual prop management interface
5. **Settings Configuration** - Complete API and preference management
6. **Image Display System** - Inline conversation images with gallery

## Handoff to Phase 5
Phase 4 completes the core feature set and provides Phase 5 with:
- Full multimedia capabilities for final polish
- Complete agent ecosystem for comprehensive testing
- Rich user interface components for UX optimization
- Advanced features for user acceptance testing

The system is now feature-complete and ready for comprehensive testing, performance optimization, and deployment preparation in Phase 5.
