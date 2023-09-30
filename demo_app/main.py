"""Python file to serve as the frontend"""
from langchain import OpenAI # LLMMathChain SerpAPIWrapper?
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent, ZeroShotAgent, Tool, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.chains import LLMChain
import os
import chainlit as cl


def create_chain(llm, template_str, memory):
    prompt = PromptTemplate(input_variables=["input", "game_state"], template=template_str)
    readonly_memory = ReadOnlySharedMemory(memory=memory)
    return LLMChain(llm=llm, prompt=prompt, verbose=True, memory=readonly_memory)

@cl.on_chat_start
def start():
    memory = cl.user_session.get("memory") or ConversationBufferMemory(memory_key="game_state")
    llmc = ChatOpenAI(temperature=0, streaming=True, model="gpt-3.5-turbo")
    search = GoogleSearchAPIWrapper()

    tools_data = {
        "Campaign State Tracker": {
            "template": """The player just said "{input}".  Summarize the updated state of the game after the player's turn.\n\nCampaign Summary: {game_state}""",
            "description": "Summarizes the current state of the D&D campaign, incorporating the latest player actions. Useful for maintaining a coherent game state."
        },
        "Campaign Start": {
            "template": """A new D&D is about to begin, get the ball rolling.  As Dungeon Master for this D&D this campaign, what should happen next after a player says "{input}"?\n\nCampaign Summary: {game_state}""",
            "description": "Initiates the start of a new D&D campaign, setting the scene and introducing initial plot elements. Useful for when a new campaign or session is starting."
        },
        "Dungeon Master Thoughts": {
            "template": """As Dungeon Master for the current D&D 5e campaign, I need to decide what should happen next after a player says "{input}".\n\nCampaign Summary: {game_state}""",
            "description": "Generates the Dungeon Master's internal thoughts about the ongoing campaign, offering insights into possible future developments. Useful for internal planning and decision-making."
        },
        "Dungeon Master Reply": {
            "template": """As Dungeon Master for the current D&D 5e campaign, what should I reply to a player that just said "{input}"?\n\nCampaign Summary: {game_state}""",
            "description": "Generates the Dungeon Master's immediate spoken reply to a player's action or query. Useful for real-time interactions with players."
        },
        "Dice Check": {
            "template": """As Dungeon Master for the current D&D 5e campaign, you should ask for a dice check from the player after they say "{input}".  You may choose to reveal the DC for the roll or not.\n\nCampaign Summary: {game_state}""",
            "description": "Determines whether a dice check is needed in response to a player's action, and what the parameters of that check should be. Useful for integrating gameplay mechanics into the narrative."
        },
    }

    templates = []
    descriptions = []
    tools = []
    for tool_name, tool_info in tools_data.items():
        template_str = tool_info["template"]
        description = tool_info["description"]

        chain = create_chain(llmc, template_str, memory)
        tool = Tool(name=tool_name, func=chain.run, description=description)
        tools.append(tool)

    def roll_dice(x: int, y: int) -> int:
        return sum([random.randint(1, y) for _ in range(x)])

    # Add more tools as needed
    tools.append(Tool(name="Dungeons and Dragons Reference", func=search.run, description="useful for answering D&D questions"))

    prefix = """Reply to the player's message directly as this one person one shot campaign's Dungeon Master...

    Player's message: {input}
    """
    suffix = """You are a Dungeon Master replying to a player's message, considering it's affect on the campaign state, you want to riff with the player...

    Player's message: {input}
    Campaign State: {game_state}

    The Dungeon Master daydreams about what might happen next in the campaign...

    {agent_scratchpad}
    """
    # Dungeon Master Scratchpad: {agent_scratchpad}

    prompt = ZeroShotAgent.create_prompt(tools, prefix=prefix, suffix=suffix, input_variables=["input", "game_state", "agent_scratchpad"])
    llm_chain = LLMChain(llm=llmc, prompt=prompt, verbose=True)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors="As a Dungeon Master, I'll put it a different way...",
        max_iterations=10,
        early_stopping_method="generate",
    )

    cl.user_session.set("agent", agent_chain)
    cl.user_session.set("memory", memory)

@cl.on_message
async def main(message: str):
    agent = cl.user_session.get("agent")
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    reply = await cl.make_async(agent.run)(message, callbacks=[cb])
    print(f"Reply: {reply}")
