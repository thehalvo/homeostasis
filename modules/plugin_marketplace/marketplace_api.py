"""
Plugin Marketplace API

This module provides the REST API for the USHS plugin marketplace,
enabling plugin discovery, download, publishing, and management.
"""

import hashlib
import json
import logging
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import jwt
import redis
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer,
                        String, Text, create_engine)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func
from werkzeug.utils import secure_filename

from .plugin_discovery import PluginManifest, PluginRegistry
from .plugin_security import PluginSecurityManager, SecurityLevel
from .plugin_storage import PluginStorage

logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()


class Plugin(Base):
    """Plugin model for marketplace database."""

    __tablename__ = "plugins"

    id = Column(String(255), primary_key=True)  # name@version
    name = Column(String(100), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    type = Column(String(50), nullable=False, index=True)
    display_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    author_id = Column(String(255), ForeignKey("authors.id"), nullable=False)
    license = Column(String(50), nullable=False)
    homepage = Column(String(500))
    repository = Column(String(500))
    keywords = Column(Text)  # JSON array
    icon_url = Column(String(500))

    # Marketplace metadata
    downloads = Column(Integer, default=0)
    rating = Column(Float, default=0.0)
    rating_count = Column(Integer, default=0)
    pricing = Column(String(20), default="free")  # free, paid, freemium
    price = Column(Float, default=0.0)
    currency = Column(String(3), default="USD")

    # Technical metadata
    manifest_hash = Column(String(64), nullable=False)
    package_hash = Column(String(64), nullable=False)
    package_size = Column(Integer, nullable=False)
    min_ushs_version = Column(String(20), nullable=False)

    # Status
    status = Column(
        String(20), default="pending"
    )  # pending, approved, rejected, deprecated
    published_at = Column(DateTime)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_at = Column(DateTime, default=func.now())

    # Relationships
    author = relationship("Author", back_populates="plugins")
    versions = relationship("PluginVersion", back_populates="plugin")
    reviews = relationship("PluginReview", back_populates="plugin")


class Author(Base):
    """Author model for plugin developers."""

    __tablename__ = "authors"

    id = Column(String(255), primary_key=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, unique=True, index=True)
    website = Column(String(500))
    verified = Column(Boolean, default=False)
    reputation = Column(Integer, default=0)

    # GPG key for code signing
    gpg_fingerprint = Column(String(40))
    gpg_public_key = Column(Text)

    created_at = Column(DateTime, default=func.now())

    # Relationships
    plugins = relationship("Plugin", back_populates="author")


class PluginVersion(Base):
    """Plugin version history."""

    __tablename__ = "plugin_versions"

    id = Column(Integer, primary_key=True)
    plugin_id = Column(String(255), ForeignKey("plugins.id"), nullable=False)
    version = Column(String(50), nullable=False)
    changelog = Column(Text)
    package_hash = Column(String(64), nullable=False)
    released_at = Column(DateTime, default=func.now())
    deprecated = Column(Boolean, default=False)

    plugin = relationship("Plugin", back_populates="versions")


class PluginReview(Base):
    """Plugin reviews and ratings."""

    __tablename__ = "plugin_reviews"

    id = Column(Integer, primary_key=True)
    plugin_id = Column(String(255), ForeignKey("plugins.id"), nullable=False)
    author_id = Column(String(255), ForeignKey("authors.id"), nullable=False)
    rating = Column(Integer, nullable=False)  # 1-5
    title = Column(String(255))
    comment = Column(Text)
    helpful_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    plugin = relationship("Plugin", back_populates="reviews")
    author = relationship("Author")


class MarketplaceAPI:
    """REST API for the plugin marketplace."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize marketplace API.

        Args:
            config: API configuration
        """
        self.config = config
        self.app = Flask(__name__)
        CORS(self.app)

        # Initialize components
        self.registry = PluginRegistry(config.get("ushs_version", "1.0.0"))
        self.security_manager = PluginSecurityManager(
            SecurityLevel(config.get("security_level", "standard"))
        )
        self.storage = PluginStorage(
            config.get("storage_path", "/var/lib/homeostasis/plugins")
        )

        # Setup database
        self.engine = create_engine(
            config.get("database_url", "sqlite:///marketplace.db")
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # Setup cache
        self.cache = redis.Redis(
            host=config.get("redis_host", "localhost"),
            port=config.get("redis_port", 6379),
            decode_responses=True,
        )

        # Register routes
        self._register_routes()

    def _register_routes(self):
        """Register API routes."""
        # Plugin discovery
        self.app.route("/api/v1/plugins", methods=["GET"])(self.list_plugins)
        self.app.route("/api/v1/plugins/<plugin_id>", methods=["GET"])(self.get_plugin)
        self.app.route("/api/v1/plugins/search", methods=["GET"])(self.search_plugins)

        # Plugin download
        self.app.route("/api/v1/plugins/<plugin_id>/download", methods=["GET"])(
            self.download_plugin
        )

        # Plugin publishing
        self.app.route("/api/v1/plugins", methods=["POST"])(self.publish_plugin)
        self.app.route("/api/v1/plugins/<plugin_id>", methods=["PUT"])(
            self.update_plugin
        )
        self.app.route("/api/v1/plugins/<plugin_id>", methods=["DELETE"])(
            self.unpublish_plugin
        )

        # Reviews
        self.app.route("/api/v1/plugins/<plugin_id>/reviews", methods=["GET"])(
            self.list_reviews
        )
        self.app.route("/api/v1/plugins/<plugin_id>/reviews", methods=["POST"])(
            self.submit_review
        )

        # Authors
        self.app.route("/api/v1/authors/<author_id>", methods=["GET"])(self.get_author)
        self.app.route("/api/v1/authors", methods=["POST"])(self.register_author)

        # Statistics
        self.app.route("/api/v1/stats", methods=["GET"])(self.get_statistics)

        # Health check
        self.app.route("/health", methods=["GET"])(self.health_check)

    def list_plugins(self):
        """List all approved plugins."""
        try:
            # Get query parameters
            page = int(request.args.get("page", 1))
            per_page = int(request.args.get("per_page", 20))
            sort_by = request.args.get("sort_by", "downloads")
            plugin_type = request.args.get("type")

            # Create database session
            session = self.Session()

            # Build query
            query = session.query(Plugin).filter(Plugin.status == "approved")

            if plugin_type:
                query = query.filter(Plugin.type == plugin_type)

            # Apply sorting
            if sort_by == "downloads":
                query = query.order_by(Plugin.downloads.desc())
            elif sort_by == "rating":
                query = query.order_by(Plugin.rating.desc())
            elif sort_by == "recent":
                query = query.order_by(Plugin.published_at.desc())
            elif sort_by == "name":
                query = query.order_by(Plugin.display_name)

            # Paginate
            total = query.count()
            plugins = query.offset((page - 1) * per_page).limit(per_page).all()

            # Format response
            response = {
                "plugins": [self._plugin_to_dict(p) for p in plugins],
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total": total,
                    "pages": (total + per_page - 1) // per_page,
                },
            }

            session.close()
            return jsonify(response)

        except Exception as e:
            logger.error(f"Failed to list plugins: {e}")
            return jsonify({"error": str(e)}), 500

    def get_plugin(self, plugin_id: str):
        """Get plugin details."""
        try:
            # Check cache
            cached = self.cache.get(f"plugin:{plugin_id}")
            if cached:
                return jsonify(json.loads(cached))

            # Get from database
            session = self.Session()
            plugin = session.query(Plugin).filter(Plugin.id == plugin_id).first()

            if not plugin:
                return jsonify({"error": "Plugin not found"}), 404

            # Get additional details
            versions = (
                session.query(PluginVersion)
                .filter(PluginVersion.plugin_id == plugin_id)
                .order_by(PluginVersion.released_at.desc())
                .all()
            )

            # Format response
            response = self._plugin_to_dict(plugin)
            response["versions"] = [
                {
                    "version": v.version,
                    "released_at": v.released_at.isoformat(),
                    "changelog": v.changelog,
                    "deprecated": v.deprecated,
                }
                for v in versions
            ]

            # Load manifest for detailed info
            manifest_path = self.storage.get_plugin_path(plugin_id) / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, "r") as f:
                    manifest_data = json.load(f)
                response["manifest"] = manifest_data

            # Cache response
            self.cache.setex(
                f"plugin:{plugin_id}", 300, json.dumps(response)  # 5 minutes
            )

            session.close()
            return jsonify(response)

        except Exception as e:
            logger.error(f"Failed to get plugin {plugin_id}: {e}")
            return jsonify({"error": str(e)}), 500

    def search_plugins(self):
        """Search plugins."""
        try:
            # Get search parameters
            query_str = request.args.get("q", "")
            plugin_type = request.args.get("type")
            capabilities = request.args.getlist("capability")
            min_rating = float(request.args.get("min_rating", 0))
            pricing = request.args.get("pricing")

            # Create database session
            session = self.Session()

            # Build search query
            query = session.query(Plugin).filter(Plugin.status == "approved")

            if query_str:
                search_pattern = f"%{query_str}%"
                query = query.filter(
                    Plugin.display_name.ilike(search_pattern)
                    | Plugin.description.ilike(search_pattern)
                    | Plugin.keywords.ilike(search_pattern)
                )

            if plugin_type:
                query = query.filter(Plugin.type == plugin_type)

            if min_rating > 0:
                query = query.filter(Plugin.rating >= min_rating)

            if pricing:
                query = query.filter(Plugin.pricing == pricing)

            # Apply capability filter (requires loading manifests)
            results = query.all()
            if capabilities:
                filtered_results = []
                for plugin in results:
                    manifest_path = (
                        self.storage.get_plugin_path(plugin.id) / "manifest.json"
                    )
                    if manifest_path.exists():
                        with open(manifest_path, "r") as f:
                            manifest = json.load(f)

                        plugin_caps = set(
                            manifest.get("capabilities", {}).get("required", [])
                        )
                        plugin_caps.update(
                            manifest.get("capabilities", {}).get("optional", [])
                        )

                        if all(cap in plugin_caps for cap in capabilities):
                            filtered_results.append(plugin)

                results = filtered_results

            # Format response
            response = {
                "results": [self._plugin_to_dict(p) for p in results],
                "total": len(results),
                "query": {
                    "q": query_str,
                    "type": plugin_type,
                    "capabilities": capabilities,
                    "min_rating": min_rating,
                    "pricing": pricing,
                },
            }

            session.close()
            return jsonify(response)

        except Exception as e:
            logger.error(f"Failed to search plugins: {e}")
            return jsonify({"error": str(e)}), 500

    def download_plugin(self, plugin_id: str):
        """Download plugin package."""
        try:
            # Get plugin info
            session = self.Session()
            plugin = session.query(Plugin).filter(Plugin.id == plugin_id).first()

            if not plugin:
                return jsonify({"error": "Plugin not found"}), 404

            if plugin.status != "approved":
                return jsonify({"error": "Plugin not available for download"}), 403

            # Update download count
            plugin.downloads += 1
            session.commit()

            # Get package file
            package_path = self.storage.get_package_path(plugin_id)
            if not package_path.exists():
                return jsonify({"error": "Package file not found"}), 404

            session.close()

            # Send file
            return send_file(
                package_path,
                as_attachment=True,
                download_name=f"{plugin.name}-{plugin.version}.tar.gz",
            )

        except Exception as e:
            logger.error(f"Failed to download plugin {plugin_id}: {e}")
            return jsonify({"error": str(e)}), 500

    def publish_plugin(self):
        """Publish a new plugin."""
        try:
            # Check authentication
            author_id = self._authenticate_request()
            if not author_id:
                return jsonify({"error": "Authentication required"}), 401

            # Check if plugin package is uploaded
            if "package" not in request.files:
                return jsonify({"error": "No package file uploaded"}), 400

            package_file = request.files["package"]
            if package_file.filename == "":
                return jsonify({"error": "No package file selected"}), 400

            # Save uploaded file temporarily
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                package_path = temp_path / secure_filename(package_file.filename)
                package_file.save(package_path)

                # Extract and validate
                extract_path = temp_path / "extracted"
                shutil.unpack_archive(package_path, extract_path)

                # Find manifest
                manifest_files = list(extract_path.glob("**/manifest.json"))
                if not manifest_files:
                    return jsonify({"error": "No manifest.json found in package"}), 400

                # Load and validate manifest
                with open(manifest_files[0], "r") as f:
                    manifest_data = json.load(f)

                try:
                    manifest = PluginManifest(manifest_data)
                except ValueError as e:
                    return jsonify({"error": f"Invalid manifest: {e}"}), 400

                # Check if plugin already exists
                plugin_id = f"{manifest.name}@{manifest.version}"
                session = self.Session()
                existing = session.query(Plugin).filter(Plugin.id == plugin_id).first()

                if existing:
                    return jsonify({"error": "Plugin version already exists"}), 409

                # Security validation
                plugin_dir = manifest_files[0].parent
                is_secure, security_issues = (
                    self.security_manager.validate_plugin_security(plugin_dir)
                )

                if not is_secure and self.config.get("enforce_security", True):
                    return (
                        jsonify(
                            {
                                "error": "Security validation failed",
                                "issues": security_issues,
                            }
                        ),
                        400,
                    )

                # Calculate hashes
                manifest_hash = hashlib.sha256(
                    json.dumps(manifest_data, sort_keys=True).encode()
                ).hexdigest()

                package_hash = hashlib.sha256(package_path.read_bytes()).hexdigest()

                # Create plugin record
                plugin = Plugin(
                    id=plugin_id,
                    name=manifest.name,
                    version=manifest.version,
                    type=manifest.type.value,
                    display_name=manifest.display_name,
                    description=manifest.description,
                    author_id=author_id,
                    license=manifest_data.get("license", "Unknown"),
                    homepage=manifest_data.get("homepage"),
                    repository=manifest_data.get("repository", {}).get("url"),
                    keywords=json.dumps(manifest_data.get("keywords", [])),
                    icon_url=manifest_data.get("icon"),
                    manifest_hash=manifest_hash,
                    package_hash=package_hash,
                    package_size=package_path.stat().st_size,
                    min_ushs_version=manifest_data.get("engines", {}).get("ushs", "*"),
                    status=(
                        "pending"
                        if self.config.get("review_required", True)
                        else "approved"
                    ),
                )

                # Set marketplace metadata if provided
                marketplace = manifest_data.get("marketplace", {})
                plugin.pricing = marketplace.get("pricing", "free")
                if "price" in marketplace:
                    plugin.price = marketplace["price"].get("amount", 0)
                    plugin.currency = marketplace["price"].get("currency", "USD")

                session.add(plugin)

                # Create version record
                version = PluginVersion(
                    plugin_id=plugin_id,
                    version=manifest.version,
                    changelog=request.form.get("changelog", ""),
                    package_hash=package_hash,
                )
                session.add(version)

                # Store plugin files
                self.storage.store_plugin(plugin_id, plugin_dir, package_path)

                # Commit to database
                session.commit()

                # Clear cache
                self.cache.delete(f"plugin:{plugin_id}")

                response = {
                    "plugin_id": plugin_id,
                    "status": plugin.status,
                    "message": "Plugin published successfully",
                }

                if security_issues:
                    response["warnings"] = security_issues

                session.close()
                return jsonify(response), 201

        except Exception as e:
            logger.error(f"Failed to publish plugin: {e}")
            return jsonify({"error": str(e)}), 500

    def update_plugin(self, plugin_id: str):
        """Update plugin metadata."""
        try:
            # Check authentication
            author_id = self._authenticate_request()
            if not author_id:
                return jsonify({"error": "Authentication required"}), 401

            # Get plugin
            session = self.Session()
            plugin = session.query(Plugin).filter(Plugin.id == plugin_id).first()

            if not plugin:
                return jsonify({"error": "Plugin not found"}), 404

            if plugin.author_id != author_id:
                return jsonify({"error": "Not authorized to update this plugin"}), 403

            # Update allowed fields
            data = request.get_json()
            allowed_fields = ["description", "homepage", "keywords", "icon_url"]

            for field in allowed_fields:
                if field in data:
                    if field == "keywords":
                        setattr(plugin, field, json.dumps(data[field]))
                    else:
                        setattr(plugin, field, data[field])

            # Update marketplace metadata
            if "marketplace" in data:
                marketplace = data["marketplace"]
                if "pricing" in marketplace:
                    plugin.pricing = marketplace["pricing"]
                if "price" in marketplace:
                    plugin.price = marketplace["price"].get("amount", plugin.price)
                    plugin.currency = marketplace["price"].get(
                        "currency", plugin.currency
                    )

            session.commit()

            # Clear cache
            self.cache.delete(f"plugin:{plugin_id}")

            session.close()
            return jsonify({"message": "Plugin updated successfully"})

        except Exception as e:
            logger.error(f"Failed to update plugin {plugin_id}: {e}")
            return jsonify({"error": str(e)}), 500

    def unpublish_plugin(self, plugin_id: str):
        """Unpublish a plugin."""
        try:
            # Check authentication
            author_id = self._authenticate_request()
            if not author_id:
                return jsonify({"error": "Authentication required"}), 401

            # Get plugin
            session = self.Session()
            plugin = session.query(Plugin).filter(Plugin.id == plugin_id).first()

            if not plugin:
                return jsonify({"error": "Plugin not found"}), 404

            if plugin.author_id != author_id:
                return (
                    jsonify({"error": "Not authorized to unpublish this plugin"}),
                    403,
                )

            # Mark as deprecated
            plugin.status = "deprecated"
            session.commit()

            # Clear cache
            self.cache.delete(f"plugin:{plugin_id}")

            session.close()
            return jsonify({"message": "Plugin unpublished successfully"})

        except Exception as e:
            logger.error(f"Failed to unpublish plugin {plugin_id}: {e}")
            return jsonify({"error": str(e)}), 500

    def list_reviews(self, plugin_id: str):
        """List plugin reviews."""
        try:
            page = int(request.args.get("page", 1))
            per_page = int(request.args.get("per_page", 10))

            session = self.Session()

            # Get reviews
            query = (
                session.query(PluginReview)
                .filter(PluginReview.plugin_id == plugin_id)
                .order_by(
                    PluginReview.helpful_count.desc(), PluginReview.created_at.desc()
                )
            )

            total = query.count()
            reviews = query.offset((page - 1) * per_page).limit(per_page).all()

            # Format response
            response = {
                "reviews": [
                    {
                        "id": r.id,
                        "author": {"id": r.author.id, "name": r.author.name},
                        "rating": r.rating,
                        "title": r.title,
                        "comment": r.comment,
                        "helpful_count": r.helpful_count,
                        "created_at": r.created_at.isoformat(),
                        "updated_at": r.updated_at.isoformat(),
                    }
                    for r in reviews
                ],
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total": total,
                    "pages": (total + per_page - 1) // per_page,
                },
            }

            session.close()
            return jsonify(response)

        except Exception as e:
            logger.error(f"Failed to list reviews for {plugin_id}: {e}")
            return jsonify({"error": str(e)}), 500

    def submit_review(self, plugin_id: str):
        """Submit a plugin review."""
        try:
            # Check authentication
            author_id = self._authenticate_request()
            if not author_id:
                return jsonify({"error": "Authentication required"}), 401

            data = request.get_json()

            # Validate input
            rating = data.get("rating")
            if not rating or rating < 1 or rating > 5:
                return jsonify({"error": "Invalid rating (must be 1-5)"}), 400

            session = self.Session()

            # Check if plugin exists
            plugin = session.query(Plugin).filter(Plugin.id == plugin_id).first()
            if not plugin:
                return jsonify({"error": "Plugin not found"}), 404

            # Check if user already reviewed
            existing = (
                session.query(PluginReview)
                .filter(
                    PluginReview.plugin_id == plugin_id,
                    PluginReview.author_id == author_id,
                )
                .first()
            )

            if existing:
                # Update existing review
                existing.rating = rating
                existing.title = data.get("title", existing.title)
                existing.comment = data.get("comment", existing.comment)
            else:
                # Create new review
                review = PluginReview(
                    plugin_id=plugin_id,
                    author_id=author_id,
                    rating=rating,
                    title=data.get("title"),
                    comment=data.get("comment"),
                )
                session.add(review)

            # Update plugin rating
            avg_rating = (
                session.query(func.avg(PluginReview.rating))
                .filter(PluginReview.plugin_id == plugin_id)
                .scalar()
            )

            rating_count = (
                session.query(func.count(PluginReview.id))
                .filter(PluginReview.plugin_id == plugin_id)
                .scalar()
            )

            plugin.rating = float(avg_rating or 0)
            plugin.rating_count = rating_count

            session.commit()

            # Clear cache
            self.cache.delete(f"plugin:{plugin_id}")

            session.close()
            return jsonify({"message": "Review submitted successfully"}), 201

        except Exception as e:
            logger.error(f"Failed to submit review: {e}")
            return jsonify({"error": str(e)}), 500

    def get_author(self, author_id: str):
        """Get author profile."""
        try:
            session = self.Session()
            author = session.query(Author).filter(Author.id == author_id).first()

            if not author:
                return jsonify({"error": "Author not found"}), 404

            # Get author's plugins
            plugins = (
                session.query(Plugin)
                .filter(Plugin.author_id == author_id, Plugin.status == "approved")
                .all()
            )

            response = {
                "id": author.id,
                "name": author.name,
                "website": author.website,
                "verified": author.verified,
                "reputation": author.reputation,
                "plugins": [
                    {
                        "id": p.id,
                        "name": p.name,
                        "display_name": p.display_name,
                        "type": p.type,
                        "downloads": p.downloads,
                        "rating": p.rating,
                    }
                    for p in plugins
                ],
                "member_since": author.created_at.isoformat(),
            }

            session.close()
            return jsonify(response)

        except Exception as e:
            logger.error(f"Failed to get author {author_id}: {e}")
            return jsonify({"error": str(e)}), 500

    def register_author(self):
        """Register a new author."""
        try:
            data = request.get_json()

            # Validate input
            required_fields = ["name", "email"]
            for field in required_fields:
                if field not in data:
                    return jsonify({"error": f"Missing required field: {field}"}), 400

            session = self.Session()

            # Check if email already exists
            existing = (
                session.query(Author).filter(Author.email == data["email"]).first()
            )
            if existing:
                return jsonify({"error": "Email already registered"}), 409

            # Create author
            author_id = hashlib.sha256(data["email"].encode()).hexdigest()[:16]
            author = Author(
                id=author_id,
                name=data["name"],
                email=data["email"],
                website=data.get("website"),
            )

            # Handle GPG key if provided
            if "gpg_public_key" in data:
                # Import and validate key
                try:
                    fingerprint = self.security_manager.signer.import_public_key(
                        data["gpg_public_key"]
                    )
                    author.gpg_fingerprint = fingerprint
                    author.gpg_public_key = data["gpg_public_key"]
                except Exception as e:
                    logger.warning(f"Failed to import GPG key: {e}")

            session.add(author)
            session.commit()

            # Generate API token
            token = self._generate_token(author_id)

            response = {
                "author_id": author_id,
                "token": token,
                "message": "Author registered successfully",
            }

            session.close()
            return jsonify(response), 201

        except Exception as e:
            logger.error(f"Failed to register author: {e}")
            return jsonify({"error": str(e)}), 500

    def get_statistics(self):
        """Get marketplace statistics."""
        try:
            session = self.Session()

            stats = {
                "total_plugins": session.query(Plugin)
                .filter(Plugin.status == "approved")
                .count(),
                "total_downloads": session.query(func.sum(Plugin.downloads)).scalar()
                or 0,
                "total_authors": session.query(Author).count(),
                "plugins_by_type": {},
                "top_plugins": [],
                "recent_plugins": [],
            }

            # Plugins by type
            for plugin_type in [
                "language",
                "analysis",
                "integration",
                "deployment",
                "monitoring",
            ]:
                count = (
                    session.query(Plugin)
                    .filter(Plugin.type == plugin_type, Plugin.status == "approved")
                    .count()
                )
                stats["plugins_by_type"][plugin_type] = count

            # Top plugins
            top_plugins = (
                session.query(Plugin)
                .filter(Plugin.status == "approved")
                .order_by(Plugin.downloads.desc())
                .limit(10)
                .all()
            )

            stats["top_plugins"] = [
                {
                    "id": p.id,
                    "name": p.display_name,
                    "downloads": p.downloads,
                    "rating": p.rating,
                }
                for p in top_plugins
            ]

            # Recent plugins
            recent_plugins = (
                session.query(Plugin)
                .filter(Plugin.status == "approved")
                .order_by(Plugin.published_at.desc())
                .limit(10)
                .all()
            )

            stats["recent_plugins"] = [
                {
                    "id": p.id,
                    "name": p.display_name,
                    "published_at": (
                        p.published_at.isoformat() if p.published_at else None
                    ),
                }
                for p in recent_plugins
            ]

            session.close()
            return jsonify(stats)

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return jsonify({"error": str(e)}), 500

    def health_check(self):
        """Health check endpoint."""
        try:
            # Check database
            session = self.Session()
            session.execute("SELECT 1")
            session.close()

            # Check cache
            self.cache.ping()

            return jsonify(
                {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
            )

        except Exception as e:
            return (
                jsonify(
                    {
                        "status": "unhealthy",
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                ),
                503,
            )

    def _authenticate_request(self) -> Optional[str]:
        """Authenticate API request."""
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        token = auth_header.split(" ")[1]

        try:
            payload = jwt.decode(token, self.config["jwt_secret"], algorithms=["HS256"])
            return payload.get("author_id")
        except jwt.InvalidTokenError:
            return None

    def _generate_token(self, author_id: str) -> str:
        """Generate JWT token for author."""
        payload = {
            "author_id": author_id,
            "exp": datetime.utcnow() + timedelta(days=30),
        }

        return jwt.encode(payload, self.config["jwt_secret"], algorithm="HS256")

    def _plugin_to_dict(self, plugin: Plugin) -> Dict[str, Any]:
        """Convert plugin model to dictionary."""
        return {
            "id": plugin.id,
            "name": plugin.name,
            "version": plugin.version,
            "type": plugin.type,
            "display_name": plugin.display_name,
            "description": plugin.description,
            "author": {
                "id": plugin.author.id,
                "name": plugin.author.name,
                "verified": plugin.author.verified,
            },
            "license": plugin.license,
            "homepage": plugin.homepage,
            "repository": plugin.repository,
            "keywords": json.loads(plugin.keywords) if plugin.keywords else [],
            "icon_url": plugin.icon_url,
            "downloads": plugin.downloads,
            "rating": plugin.rating,
            "rating_count": plugin.rating_count,
            "pricing": plugin.pricing,
            "price": plugin.price if plugin.pricing != "free" else None,
            "currency": plugin.currency if plugin.pricing != "free" else None,
            "published_at": (
                plugin.published_at.isoformat() if plugin.published_at else None
            ),
            "updated_at": plugin.updated_at.isoformat(),
        }

    def run(self, host: str = "127.0.0.1", port: int = 5000, debug: bool = False):
        """Run the API server."""
        self.app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    # Example configuration
    config = {
        "database_url": "sqlite:///marketplace.db",
        "redis_host": "localhost",
        "redis_port": 6379,
        "storage_path": "/var/lib/homeostasis/plugins",
        "jwt_secret": "your-secret-key-here",
        "ushs_version": "1.0.0",
        "security_level": "standard",
        "enforce_security": True,
        "review_required": True,
    }

    # Create and run API
    api = MarketplaceAPI(config)
    api.run(debug=False)
