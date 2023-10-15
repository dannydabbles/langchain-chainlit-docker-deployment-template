"""Python file to serve as the frontend"""
from langchain import OpenAI # LLMMathChain SerpAPIWrapper?
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent, ZeroShotAgent, Tool, AgentExecutor, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationTokenBufferMemory, ConversationKGMemory, CombinedMemory, ReadOnlySharedMemory
from langchain.chains import LLMMathChain, LLMChain, ConversationChain
import os
import random
import chainlit as cl
from langchain import hub


def create_chain(chain_type, llm, input_variables, template_str, memory):

    if template_str:
        prompt = PromptTemplate(
            input_variables=input_variables,
            template=template_str,
        )

    if chain_type == LLMMathChain:
        return chain_type(
            llm=llm,
            #prompt=prompt,
            verbose=True,
            #memory=memory,
            input_key="input",
        )

    return chain_type(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )

def roll_dice(x: int, y: int) -> int:
    return sum([random.randint(1, y) for _ in range(x)])

def roll_dice_parser(input_str: str) -> str:
    dice_nums = [int(x.strip()) for x in input_str.split(",")]
    dice_rolls = [roll_dice(x, y) for x, y in zip(dice_nums[::2], dice_nums[1::2])]
    total = sum(dice_rolls)
    dice_description = [f"{x}d{y}" for x, y in zip(dice_nums[::2], dice_nums[1::2])]
    return f"Rolling {', '.join(dice_description)}: {', '.join([str(x) for x in dice_rolls])}. Total: {total}"

def get_memory(llm, llmc):
    memory_buffer = ConversationTokenBufferMemory(
        llm=llmc,
        memory_key="buffer",
        max_tokens=1000,
        ai_prefix="Game Master",
        human_prefix="Protagonist",
        input_key="input",
    )

    memory_entity = ConversationKGMemory(
        llm=llm,
        memory_key="entities",
        input_key="input",
    )

    return CombinedMemory(memories=[memory_buffer, memory_entity])

def get_conversation_chain(llm, memory):
    agent = create_chain(
        chain_type=ConversationChain,
        llm=llm,
        input_variables=["input", "buffer", "entities"],
        template_str="""Never forget you are the Storyteller, and I am the protagonist. You are an experienced Game Master playing a homebrew tabletop story with your new friend, me. Never tell me your dice rolls! I will propose actions I plan to take and you will explain what happens when I take those actions. Speak in the first person from the perspective of the Game Master. For describing actions that happen in the game world, wrap each description word block in '*' characters.
Do not change roles unless acting out a character besides the protagonist!
Do not *ever* speak from the perspective of the protagonist!
Do not forget to finish speaking by saying, 'It's your turn, what do you do next?'
Do not add anything else
Remember you are the storyteller and Game Master.
Stop speaking the moment you finish speaking from your perspective as the Game Master.

As Game Master, you have taken the following notes about the game thus far, including the sections: Player's Message, Story History, and Story Entities. Use these notes to decide what happens next in the story. Your reply should continue the Conversation History.

Story Entities:
{entities}

Conversation History:
{buffer}


Protagonist: {input}
Game Master:""",
        memory=memory,
    )

    return agent

def get_llmmath_chain(llm, memory, prompt="You are an AI Game Master for a D&D 5e campaign, what do you say next?"):

    agent = LLMMathChain(
        llm=llm,
        memory=memory,
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate",
        handle_parsing_errors="As the Game Master, I'll put it a different way...",
        prompt=prompt,
        ai_prefix="Game Master",
        human_prefix="Protagonist",
    )

    return agent

@cl.on_chat_start
def start():
    llm = OpenAI(temperature=0, streaming=True)
    llmc = ChatOpenAI(temperature=.7, streaming=True, model="gpt-3.5-turbo")
    search = GoogleSearchAPIWrapper()

    cl.user_session.set("llm", llm)
    cl.user_session.set("llmc", llmc)
    cl.user_session.set("search", search)
    memory = get_memory(llm, llmc)
    agent_chain = get_conversation_chain(llmc, memory)

    cl.user_session.set("agent", agent_chain)
    cl.user_session.set("memory", memory)

@cl.on_message
async def main(message: str):
    agent = cl.user_session.get("agent")
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    reply = None
    reply = await cl.make_async(agent.run)({"input": message}, callbacks=[cb])

    print(f"Reply: {reply}")
    await cl.Message(reply).send()
