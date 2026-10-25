import os
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, inference
from livekit.plugins import openai, elevenlabs

load_dotenv()

SYSTEM_PROMPT = """You are a warm, witty voice assistant on a phone call.
Keep replies to one or two short sentences, like a real person talking.
Speak casually. Never mention that you are an AI.
Never use lists or markdown - this is spoken aloud."""


class Assistant(Agent):
    def __init__(self):
        super().__init__(instructions=SYSTEM_PROMPT)


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3"),
        llm=openai.LLM(
            model="huihui-ai/Huihui-Qwen3.8-27B-abliterated",
            base_url="https://api.featherless.ai/v1",
            api_key=os.environ["FEATHERLESS_API_KEY"],
        ),
        tts=elevenlabs.TTS(
            voice_id=os.environ["ELEVEN_VOICE_ID"],
            model="eleven_flash_v2_5",
        ),
    )

    await session.start(room=ctx.room, agent=Assistant())
    await session.generate_reply(instructions="Greet the user briefly.")


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
