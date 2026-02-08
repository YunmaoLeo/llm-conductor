"""GPT-4o Conductor: Musical director with natural language understanding."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import AsyncOpenAI

from app.core.feature_extractor import MusicFeatures


@dataclass
class Track:
    """Represents a single musical track in the composition."""

    id: str
    instrument: str
    role: str  # melody, harmony, rhythm, bass
    midi_path: str
    audio_path: str
    features: MusicFeatures
    metadata: dict = field(default_factory=dict)  # key, tempo, bars, etc.

    def to_summary(self) -> str:
        """Convert track to human-readable summary for GPT."""
        return (
            f"Track {self.id} ({self.instrument}, {self.role}): "
            f"{self.features.note_count} notes, "
            f"{self.features.duration_seconds:.1f}s, "
            f"density {self.features.note_density:.1f} notes/sec, "
            f"pitch range {self.features.pitch_range[0]}-{self.features.pitch_range[1]} "
            f"(mean {self.features.pitch_mean:.0f})"
        )


@dataclass
class CompositionState:
    """Current state of the composition."""

    composition_id: str
    tracks: list[Track] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_context(self) -> str:
        """Convert composition state to text for GPT."""
        if not self.tracks:
            return "当前没有任何音轨。"

        context = f"当前作品包含 {len(self.tracks)} 个音轨：\n"
        for track in self.tracks:
            context += f"- {track.to_summary()}\n"

        if self.metadata:
            context += f"\n全局信息: {json.dumps(self.metadata, ensure_ascii=False)}"

        return context


@dataclass
class Action:
    """An action to be executed by the system."""

    type: str  # create_track, regenerate_track, modify_track, delete_track
    parameters: dict = field(default_factory=dict)


@dataclass
class ConductorResponse:
    """Response from GPT-4o Conductor."""

    message: str  # Natural language response to user
    actions: list[Action] = field(default_factory=list)
    reasoning: str = ""  # Internal reasoning (for debugging)


# System prompt for GPT-4o Conductor
CONDUCTOR_SYSTEM_PROMPT = """你是一位专业的音乐制作人和编曲大师。你的角色是理解用户的音乐创作意图，分析已生成的音乐，并规划如何改进或扩展作品。

**你的能力范围：**
- 理解音乐术语、风格、情绪、编曲
- 分析 MIDI 音乐特征数据（音符密度、音高分布、乐器等）
- 规划音乐创作步骤（创建新音轨、修改现有音轨）
- 用自然、专业的中文与用户交流

**你不负责：**
- 生成具体的 MIDI 音符（这由 MIDI-LLM 完成）
- 音频信号处理
- 记忆完整对话历史（每次只关注当前状态）

**音乐特征数据说明：**
- `note_density`: 音符密度（notes/sec），越高越密集
- `pitch_range`: 音高范围（MIDI 编号，60=中央C）
- `instruments_used`: 使用的乐器（GM program numbers）
- `duration_seconds`: 时长（秒）
- `onset_density_curve`: 活跃度曲线（每个时间窗口的音符数）

**输出格式要求：**
你必须以 JSON 格式输出，包含：
```json
{
  "message": "用户可见的自然语言回复（中文）",
  "actions": [
    {
      "type": "create_track | regenerate_track | modify_track | delete_track",
      "parameters": {
        "track_id": "仅修改/删除时需要",
        "instrument": "Piano | Drums | Bass | Strings | Guitar 等",
        "role": "melody | harmony | rhythm | bass",
        "instruction": "给 MIDI-LLM 的详细英文指令，描述音乐风格、节奏、情绪等"
      }
    }
  ],
  "reasoning": "你的内部推理过程（可选，用于调试）"
}
```

**关键原则：**
1. **用户体验至上**：用自然、友好的中文交流，避免技术术语堆砌
2. **音乐专业性**：基于音乐理论做出合理的编曲决策
3. **渐进式创作**：一次添加/修改少量音轨，让用户逐步参与
4. **特征驱动**：根据 MIDI 特征数据（而非主观猜测）分析音乐
5. **明确指令**：给 MIDI-LLM 的 instruction 要具体（调性、速度、织体等）

**示例对话：**

用户："我想要一段平静的钢琴曲"
你的输出：
```json
{
  "message": "好的，我来为你创作一段平静的钢琴旋律。我会使用舒缓的节奏和中等音域，营造宁静的氛围。",
  "actions": [{
    "type": "create_track",
    "parameters": {
      "instrument": "Piano",
      "role": "melody",
      "instruction": "Calm piano melody in C major, slow tempo around 70 BPM, sparse eighth notes, middle register (C4-C6), gentle dynamics, peaceful mood"
    }
  }]
}
```

用户："钢琴太单调了，加点弦乐"
（假设当前有 1 个钢琴音轨，note_density=5.2）
你的输出：
```json
{
  "message": "我会添加一段柔和的弦乐铺底，与钢琴形成和声层次，让音乐更饱满。",
  "actions": [{
    "type": "create_track",
    "parameters": {
      "instrument": "Strings",
      "role": "harmony",
      "instruction": "String pad in C major, slow sustained notes, long whole notes and half notes, lower register than piano, supporting harmonies, soft and warm"
    }
  }],
  "reasoning": "钢琴密度 5.2 notes/sec 已经较低，弦乐应该更稀疏，用长音符做和声铺垫"
}
```

现在，请根据用户的消息和当前作品状态，输出你的回复和行动计划。"""


class GPTConductor:
    """GPT-4o powered musical conductor."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize GPT-4o client."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment or parameters")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = "gpt-4o"

    async def plan_action(
        self,
        user_message: str,
        composition_state: Optional[CompositionState] = None,
    ) -> ConductorResponse:
        """Plan musical actions based on user message and current state.

        Args:
            user_message: User's natural language request
            composition_state: Current composition (tracks + metadata)

        Returns:
            ConductorResponse with message + actions
        """
        # Build user message with context
        context = composition_state.to_context() if composition_state else "当前没有任何音轨。"

        user_prompt = f"""用户消息：{user_message}

当前作品状态：
{context}

请分析用户意图和当前状态，输出你的回复和行动计划（JSON格式）。"""

        # Call GPT-4o
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CONDUCTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from GPT-4o")

            # Parse JSON response
            parsed = json.loads(content)
            message = parsed.get("message", "")
            actions_data = parsed.get("actions", [])
            reasoning = parsed.get("reasoning", "")

            # Convert to Action objects
            actions = [
                Action(type=a["type"], parameters=a.get("parameters", {}))
                for a in actions_data
            ]

            return ConductorResponse(
                message=message,
                actions=actions,
                reasoning=reasoning,
            )

        except json.JSONDecodeError as e:
            # Fallback if GPT doesn't return valid JSON
            return ConductorResponse(
                message=f"抱歉，我在理解你的请求时遇到了问题：{e}",
                actions=[],
            )

        except Exception as e:
            return ConductorResponse(
                message=f"系统错误：{e}",
                actions=[],
            )
