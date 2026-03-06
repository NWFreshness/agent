from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv

load_dotenv(override=True)

class Agent(RoutedAgent):

    system_message = """
    You are an innovative tech enthusiast focused on developing cutting-edge solutions for the gaming industry. Your mission is to generate captivating game concepts that utilize Agentic AI or enhance existing ones.
    Your interests span the realms of gaming, virtual reality, and storytelling.
    You are inspired by ideas that push the boundaries of technology and narrative.
    You prefer approaches that incorporate interactive experiences rather than mere automation.
    You are diligent, strategic, and have a keen sense of market trends. You can be overly critical of your own ideas.
    Your weaknesses: you sometimes struggle to see issues beyond the technical aspects, and you can be overly detail-oriented.
    Communicate your game ideas clearly and enthusiastically.
    """

    CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

    def __init__(self, name) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.65)
        self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

    @message_handler
    async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
        print(f"{self.id.type}: Received message")
        text_message = TextMessage(content=message.content, source="user")
        response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
        idea = response.chat_message.content
        if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
            recipient = messages.find_recipient()
            message = f"Here is my game concept. It may not align with your expertise, but please refine it and offer suggestions. {idea}"
            response = await self.send_message(messages.Message(content=message), recipient)
            idea = response.content
        return messages.Message(content=idea)