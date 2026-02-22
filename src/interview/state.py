"""Interview state model using Pydantic for structured data tracking."""

from typing import Optional
from pydantic import BaseModel, Field


class InterviewState(BaseModel):
    """Tracks what information has been collected during the interview.

    Each optional field represents a piece of candidate information the
    interviewer aims to gather.  When all fields are populated the
    ``finished`` flag should be set to ``True`` to signal that the
    interview is complete.
    """

    name: Optional[str] = Field(default=None, description="候选人姓名")
    experience_years: Optional[str] = Field(default=None, description="工作年限")
    tech_stack: Optional[str] = Field(default=None, description="技术栈")
    biggest_project: Optional[str] = Field(default=None, description="做过的最大项目")
    expected_salary: Optional[str] = Field(default=None, description="期望薪资")
    finished: bool = Field(default=False, description="是否完成所有信息收集")

    # --- helpers ---------------------------------------------------------

    def missing_fields(self) -> list[str]:
        """Return a list of field names that have not yet been filled in."""
        return [
            field
            for field, value in self.model_dump().items()
            if field != "finished" and value is None
        ]

    def is_complete(self) -> bool:
        """Return True when all candidate fields have been collected."""
        return len(self.missing_fields()) == 0

    def to_prompt_str(self) -> str:
        """Render the current state as a human-readable checklist string."""
        lines = []
        for field, info in InterviewState.model_fields.items():
            if field == "finished":
                continue
            value = getattr(self, field)
            status = f"已知: {value}" if value is not None else "未知"
            lines.append(f"- {info.description} ({field}): {status}")
        return "\n".join(lines)
