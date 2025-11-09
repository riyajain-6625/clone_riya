import os
from pathlib import Path
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv
import gradio as gr
import requests

load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
RESUME_PATH = Path("Riya.pdf")
PROMPT_PATH = Path("prompt.md")


class ResumeClone:
    def __init__(self):
        load_dotenv()
        
        # inititalize the chatbot by loading resume and prompts
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.resume_text = self.extract_text_from_pdf()
        self.system_prompt = self.load_system_prompt()
        self.conversation_history = []
    
    def extract_text_from_pdf(self) -> str:
        # If resume file exists
        if not RESUME_PATH.exists():
            raise FileNotFoundError(f"Resume not found at {RESUME_PATH}")
        
        # extract text from PDF resume using PyPDF2
        try:
            reader = PdfReader(RESUME_PATH)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def load_system_prompt(self) -> str:
        if not PROMPT_PATH.exists():
            # Fall back prompt
            return """You are Riya Jain, answering questions about yourself based solely on your resume.
            Use first-person tone. Only answer using information from your resume.
            If information is not in your resume, say "I don't know" or "That's not in my resume"."""
        
        with open(PROMPT_PATH, "r", encoding='utf-8') as f:
            return f.read()

    def build_context_message(self) -> str:
        """Build the complete context including system prompt and resume"""
        return f"""{self.system_prompt}
        ---

        ## YOUR RESUME INFORMATION:

        {self.resume_text}

        ---

        Remember: Answer ONLY based on the resume information above. Use first-person (I, me, my).
        If the answer is not in the resume, say you don't know.
        """
    
    def chat(self, user_message: str, history: list) -> str:
        if not user_message.strip():
            return "Please ask me a question about my professional background!"

        # Build messages for OpenAI API
        messages = [
            {
                "role": "system",
                "content": self.build_context_message()
            }
        ]
        
        if history:
            messages.extend(history[-20:])  # limit history to last 20 messages
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        try:
            #call open ai chat completion
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500,
                temperature=0.7,
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}. Please check your API key and try again."
        
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
        return []

def create_gradio_interface():
    try:
        # Check if we have openai api key
        if not OPENAI_API_KEY:
            print("⚠️  WARNING: OPENAI_API_KEY not found in environment variables!")
            print("Please create a .env file with your OpenAI API key.")
        
        # Load resume data
        chatbot = ResumeClone()
        print("Resume loaded successfully!")
        print(f"📄 Resume preview: {chatbot.resume_text[:200]}...")
    except Exception as e:
        print(f"❌ Error creating Gradio interface: {e}")
        raise
    
    # create gradio frontend interface
    with gr.Blocks(
        title="Chat with Riya Jain",
    ) as interface:
        gr.Markdown(
            """
            # 💬 Chat with Riya Jain

            Hi! I'm Riya Jain, a Data Analyst. Ask me anything about my professional background,
            skills, experience, or projects. I'll answer based on my skills and experience!
            """
        )
        
        chatbot_ui = gr.Chatbot(
            height=300,
            label="Conversation with Riyas Clone",
            show_label=True,
            type="messages",
        )
        
        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Ask me about my skills, experience, projects, or education...",
                label="Your Question",
                scale=4
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
        
        with gr.Row():
            clear_btn = gr.Button("🔄 Clear Chat", variant="secondary")


        # Event handlers
        def respond(message, history):
            bot_response = chatbot.chat(message, history)
            
            # Append user message and assistant response in messages format
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": bot_response})
            return "", history

        msg_input.submit(respond, [msg_input, chatbot_ui], [msg_input, chatbot_ui])
        submit_btn.click(respond, [msg_input, chatbot_ui], [msg_input, chatbot_ui])
        clear_btn.click(lambda: chatbot.reset_conversation(), None, chatbot_ui)
            
    return interface



if __name__ == "__main__":
    print("🚀 Starting Chat with Riya Jain...")
    
    #this interface is for the frontend of the chatpage using gradio package
    interface = create_gradio_interface()
    
    print("✨ Launching Gradio interface...")

    interface.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860
    )
