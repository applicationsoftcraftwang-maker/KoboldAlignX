"""
SQLAlchemy database models.

TODO: Move from src/core/models.py and update imports
"""

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

# TODO: Import Base from src.db.base
# from src.db.base import Base


class Post:
    """
    Post model.
    
    TODO: Move from original src/core/models.py
    """
    # __tablename__ = "post"
    pass


class User:
    """
    User model.
    
    TODO: Move from original src/core/models.py
    """
    # __tablename__ = "user"
    pass


class Guidehawk:
    """
    Guidehawk model.
    
    TODO: Move from original src/core/models.py
    """
    # __tablename__ = "guidehawk"
    pass


class Reliance:
    """
    Reliance model.
    
    TODO: Move from original src/core/models.py
    """
    # __tablename__ = "reliance"
    pass
