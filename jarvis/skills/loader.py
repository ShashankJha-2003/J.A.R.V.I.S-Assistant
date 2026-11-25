"""Loads all built-in skill packs into a SkillRegistry instance."""

from __future__ import annotations

from jarvis.skills.app_automation import register_app_skills
from jarvis.skills.developer_tools import register_developer_skills
from jarvis.skills.memory_rag import register_memory_skills
from jarvis.skills.productivity import register_productivity_skills
from jarvis.skills.registry import SkillRegistry
from jarvis.skills.smart_home import register_smart_home_skills
from jarvis.skills.system_control import register_system_skills
from jarvis.skills.vision import register_vision_skills
from jarvis.skills.web_intelligence import register_web_skills


def create_default_registry() -> SkillRegistry:
    registry = SkillRegistry()
    register_system_skills(registry)
    register_app_skills(registry)
    register_web_skills(registry)
    register_productivity_skills(registry)
    register_memory_skills(registry)
    register_developer_skills(registry)
    register_vision_skills(registry)
    register_smart_home_skills(registry)
    return registry
