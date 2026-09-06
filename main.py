from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from huggingface_hub import InferenceClient
from typing import Optional, List, Dict, Literal
import os
import re
import random
import secrets

app = FastAPI(title="HireQuest VR API", description="Enhanced Interview Engine with Scoring & Full Feedback Loop")


def clean_api_response(text: str) -> str:
    if not text:
        return ""
    
    text = text.replace("**", "").replace("__", "").replace("`", "")
    
    junk_labels = [
        "Spoken_text:", "Question:", "Answer:", "Feedback:", 
        "Score:", "Emotion:", "Summary:", "Note:", "Task:"
    ]
    for label in junk_labels:
        text = text.replace(label, "")
        
    text = re.sub(r"^(Understood|Certainly|Sure|Great|Okay|Here is|I will).*?[:!.]", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\(Area:.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(Subtopic:.*?\)", "", text, flags=re.IGNORECASE)
    
    text = text.replace("\\n", " ").replace("\n", " ").replace('\"', '').strip()
    
    return text

hf_token = os.environ.get("HF_TOKEN") 
client = InferenceClient("deepseek-ai/DeepSeek-V3", token=hf_token)


SCENARIO_PROMPTS = {
    "Standard": "A formal office interview. Focus on technical skills.",
    "Pressure": "CRITICAL! The system is down and the company is losing money! Be urgent and demand fast solutions. No time for small talk.",
    "Conflict": "You are a stubborn senior colleague. You disagree with the candidate's previous methods. Challenge their choices aggressively.",
    "Simple-Client": "You are a non-technical client. You don't understand code. Ask 'How does this help me?' and avoid any technical jargon."
}

EVALUATION_RULES = {
    "Standard": "Focus on technical accuracy and proper terminology.",
    "Pressure": "Prioritize direct, fast, and efficient solutions. High marks for staying calm and decisive.",
    "Conflict": "Evaluate how well they defend their logic against your disagreement. Bonus for logic and respect.",
    "Simple-Client": "STRICT RULE: Deduct points for complex jargon. Higher scores for simple analogies and business value."
}

QUESTION_FOCUS_AREAS = [
    "Core Concepts",
    "Internal Architecture",
    "Performance Optimization",
    "Scalability",
    "Debugging and Troubleshooting",
    "Security and Reliability",
    "Data Modeling and Trade-offs",
    "Integration and API Design",
    "Caching and State Management",
    "Testing and Observability",
]

QUESTION_ANGLES = [
    "ask for a practical production scenario",
    "ask for a comparison between two approaches",
    "ask about a failure case and how to handle it",
    "ask about hidden trade-offs and limitations",
    "ask how the candidate would optimize a slow system",
    "ask how the candidate would design the solution from scratch",
    "ask how the candidate would debug an unexpected behavior",
    "ask how the candidate would explain the concept to a teammate",
]

def build_question_variation() -> Dict[str, str]:
    """Create backend-side diversity so new sessions do not reuse the same first prompt."""
    rng = random.SystemRandom()
    return {
        "token": secrets.token_hex(4),
        "focus": rng.choice(QUESTION_FOCUS_AREAS),
        "angle": rng.choice(QUESTION_ANGLES),
    }


class InterviewSession(BaseModel):
    name: str
    graduation_year: int
    years_of_experience: int
    track_name: str
    difficulty: Literal["Junior", "Mid-level", "Senior"] 
    interviewer_personality: Literal["Friendly", "Aggressive", "Formal", "Adaptive"] 
    question_mode: Literal["Manual", "Dynamic"]
    num_questions: Optional[int] = 5
    
    scenario: Literal["Standard", "Pressure", "Conflict", "Simple-Client"] = "Standard"
    current_question_count: int = 1

    @field_validator('difficulty', 'interviewer_personality', 'question_mode', mode='before')
    @classmethod
    def normalize_inputs(cls, v: str):
        if isinstance(v, str):
            v_clean = v.strip().lower()
            if v_clean in ["mid-level", "mid level", "midlevel"]:
                return "Mid-level"
            return v_clean.capitalize()
        return v

class DiscussionRequest(BaseModel):
    session: InterviewSession
    current_question: str
    user_message: str
    chat_history: List[Dict[str, str]] = Field(default_factory=list)


class QuestionRequest(BaseModel):
    session: InterviewSession
    asked_questions: List[str] = Field(default_factory=list)
    previous_scores: List[int] = Field(default_factory=list)


class AnswerSubmission(BaseModel):
    session_data: InterviewSession
    current_question: str
    user_answer: str
    discussion_context: Optional[str] = ""


class QAHistoryItem(BaseModel):
    question: str
    answer: str
    score: int


class SummaryRequest(BaseModel):
    session: InterviewSession
    history: List[QAHistoryItem]


class UserInput(BaseModel):
    session_data: InterviewSession
    user_input: str


class IdealAnswerRequest(BaseModel):
    session: InterviewSession
    questions: List[str]

def get_personality_instruction(personality: str):
    personalities = {
        "Friendly": (
            "Be supportive, warm, and encouraging. "
            "If the candidate is respectful, be positive and helpful. "
            "If the candidate uses insults, offensive language, or is disrespectful, "
            "DO NOT praise them or respond positively. "
            "Stay calm and professional. "
            "Politely state that the language is inappropriate and ask them to continue respectfully. "
            "Then continue the interview normally without emotional reaction."
        ),
        "Aggressive": "Be tough, challenging, and critical. If the score is low, show disappointment. If they ask for help, show impatience.",
        "Formal": "Be professional, cold, and strictly business-like.",
        "Adaptive": "Adjust based on score: If score > 7 be impressed, if score < 5 be stern/critical. Help only if they deserve it."
    }
    return personalities.get(personality, "Be professional.")

TTS_INSTRUCTIONS = (
    "OUTPUT FORMAT RULE: "
    "1. For 'spoken_text', provide ONLY the plain text to be spoken by TTS. No markdown, no labels."
    "2. No stage directions like [Sighs]."
)

@app.get("/")
async def root():
    return {
        "message": "HireQuest VR API is running",
        "docs": "/docs"
    }


@app.post("/analyze_intro")
async def analyze_intro(data: UserInput):
    persona_style = get_personality_instruction(data.session_data.interviewer_personality)
    
    prompt = (
        f"You are a {data.session_data.interviewer_personality} interviewer. {persona_style}\n"
        f"Candidate Name: {data.session_data.name}\n"
        f"Track: {data.session_data.track_name}, Level: {data.session_data.difficulty}.\n"
        f"Candidate said: '{data.user_input}'.\n\n"
        f"TASK:\n"
        f"1. Acknowledge their introduction briefly.\n"
        f"2. Confirm if their background matches the {data.session_data.track_name} track.\n"
        f"3. Transition smoothly to start the interview.\n\n"
        f"RULES:\n"
        f"- {TTS_INSTRUCTIONS}\n"
        f"- Output ONLY the spoken response. No 'Understood', no stars, no labels."
    )
    try:
        output = client.chat_completion([{"role": "user", "content": prompt}], max_tokens=250)
        raw_response = output.choices[0].message.content
        clean_response = clean_api_response(raw_response)
        return {"response": clean_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat_discussion")
async def chat_discussion(data: DiscussionRequest):
    current_scenario = SCENARIO_PROMPTS.get(data.session.scenario, SCENARIO_PROMPTS["Standard"])
    soft_skills_keywords = ["skill", "team", "handle", "conflict", "weakness", "strength", "pressure", "lead", "situation"]
    is_soft_skill = any(word in data.current_question.lower() for word in soft_skills_keywords)

    if is_soft_skill:
        instruction = "The candidate is discussing Soft Skills. Ask for a REAL-LIFE EXAMPLE (STAR Method)."
    else:
        instruction = "The candidate is confused about a Technical question. Give ONE small conceptual hint."

    prompt_text = (
        f"You are a Senior {data.session.track_name} Interviewer in a {data.session.scenario} scenario.\n"
        f"SCENARIO STYLE: {current_scenario}\n"
        f"Current Question: '{data.current_question}'.\n"
        f"Candidate said: '{data.user_message}'.\n\n"
        f"STRICT INSTRUCTIONS:\n"
        f"- {instruction}\n"
        f"- Stay in character for the scenario. Total response MUST be between 10 to 25 words.\n"
        f"- NO labels, NO markdown, NO internal thoughts.\n"
    )
    
    try:
        output = client.chat_completion([{"role": "user", "content": prompt_text}], max_tokens=60, temperature=0.4)
        return {"response": clean_api_response(output.choices[0].message.content)}
    except Exception as e:
        raise HTTPException(status_code=500, detail="AI Error")

@app.post("/generate_question")
async def generate_question(request: QuestionRequest):
    session = request.session
    current_scenario = SCENARIO_PROMPTS.get(session.scenario, SCENARIO_PROMPTS["Standard"])
    
    avg_score = sum(request.previous_scores) / len(request.previous_scores) if request.previous_scores else 10
    questions_count = len(request.asked_questions)
    
    failed_early = (questions_count >= 3 and avg_score < 4)
    should_finish = session.question_mode == "Dynamic" and (failed_early or questions_count >= 8)

    if should_finish:
        return {"question": "FINISH_INTERVIEW", "status": "failed" if failed_early else "completed"}

    history_text = "Previously Asked: " + " | ".join(request.asked_questions) if request.asked_questions else "None"
    variation = build_question_variation()
    
    encouragement_note = ""
    if session.interviewer_personality == "Friendly" and request.previous_scores:
        last_score = request.previous_scores[-1]
        encouragement_note = (
            f"Since you are a Friendly interviewer and the last score was {last_score}/10, "
            "start with a very brief encouraging sentence (5-10 words) about their previous answer, "
            "then ask the next question."
        )

    prompt = (
        f"You are a Senior {session.track_name} Architect interviewing a {session.difficulty} candidate.\n"
        f"CURRENT SCENARIO: {current_scenario}\n"
        f"Strict Technical Context: {session.track_name}\n"
        f"QUESTION DIVERSITY TOKEN: {variation['token']}\n"
        f"PRIMARY FOCUS AREA: {variation['focus']}\n"
        f"REQUIRED QUESTION ANGLE: {variation['angle']}\n"
        f"{history_text}\n\n"
        f"{encouragement_note}\n\n"
        f"TASK:\n"
        f"- Ask ONE deep technical question about {session.track_name}.\n"
        f"- Use the PRIMARY FOCUS AREA and REQUIRED QUESTION ANGLE above as a hard constraint.\n"
        f"- Focus can include: Core Concepts, Internal Architecture, Performance Optimization, Scalability, Security, Debugging, API Design, or Trade-offs.\n"
        f"- The question must be challenging and suitable for a {session.difficulty} level.\n"
        f"- DO NOT ask general or soft-skill questions. NO 'tell me about yourself'.\n"
        f"- DO NOT repeat any question from the list above.\n\n"
        f"OUTPUT RULE:\n"
        f"- Output ONLY the response text. No labels like 'Question:' or 'Feedback:'.\n"
        f"- DO NOT mention the QUESTION DIVERSITY TOKEN or any internal prompt metadata.\n"
        f"TASK: Generate Question #{questions_count + 1} for a {session.difficulty} candidate. "
        "Strictly follow the SCENARIO atmosphere. Do not repeat questions."
    )

    try:
        output = client.chat_completion(
            [{"role": "user", "content": prompt}], 
            max_tokens=200,
            temperature=0.8
        )
        raw_q = output.choices[0].message.content.strip()
        clean_q = clean_api_response(raw_q)
        return {"question": clean_q}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        

@app.post("/evaluate_answer")
async def evaluate_answer(data: AnswerSubmission):
    scenario_type = data.session_data.scenario
    current_eval_rule = EVALUATION_RULES.get(scenario_type, EVALUATION_RULES["Standard"])
    
    prompt = (
        f"Role: Senior {data.session_data.track_name} Technical Interviewer.\n"
        f"SCENARIO CONTEXT: {scenario_type} - {current_eval_rule}\n"
        f"Question: {data.current_question}\n"
        f"Discussion Context: {data.discussion_context}\n"
        f"Candidate Final Answer: {data.user_answer}\n\n"
        f"CRITERIA:\n"
        f"Context: Evaluate ONLY the technical accuracy of this specific answer.\n"
        f"- Match answer style to the SCENARIO ({scenario_type}).\n"
        f"- Look for technical keywords related to {data.session_data.track_name}.\n"
        f"- Score 0-4: Wrong or 'I don't know'.\n"
        f"- Score 5-7: Correct but shallow.\n"
        f"- Score 8-10: Detailed and precise.\n\n"
        f"Use ONLY these emotions: Happy, Natural, Sad, Excited, Angry, Disappointed, Confused.\n"
        f"OUTPUT FORMAT (STRICT): SCORE: <0-10> || EMOTION: <Type> || FEEDBACK: <Your direct response>\n"
        f"Return ONLY one emotion exactly as written above. Do not invent new emotions or synonyms."
    )

    try:
        output = client.chat_completion([{"role": "user", "content": prompt}], max_tokens=350)
        raw_text = output.choices[0].message.content.strip()
        
        if "||" in raw_text:
            parts = raw_text.split("||")
            score_match = re.search(r'\d+', parts[0])
            score = int(score_match.group()) if score_match else 5
            emotion_part = parts[1].split(":")[-1].strip() if ":" in parts[1] else parts[1].strip()
            raw_feedback = parts[2].split(":")[-1].strip() if ":" in parts[2] else parts[2].strip()
            return {
                "score": score, 
                "emotion": emotion_part, 
                "feedback": clean_api_response(raw_feedback)
            }
        
        return {"score": 5, "emotion": "Natural", "feedback": clean_api_response(raw_text)}
        
    except Exception as e:
        return {"score": 5, "emotion": "Natural", "feedback": "Technical error, but let's keep going."}

@app.post("/generate_summary")
async def generate_summary(req: SummaryRequest):
    history_str = ""
    total_score = sum(item.score for item in req.history)
    avg_score = total_score / len(req.history) if req.history else 0
    is_passed = "Passed" if avg_score >= 5 else "Failed"
    
    for idx, item in enumerate(req.history):
        history_str += f"Q{idx+1}: {item.question}\nA: {item.answer}\nScore: {item.score}/10\n"

    scenario_feedback = f"Ensure you mention how they handled the '{req.session.scenario}' scenario vibe."

    if req.session.interviewer_personality == "Friendly":
        task_instruction = (
            "- Start with a warm overall evaluation.\n"
            f"- {scenario_feedback}\n"
            "- For each question, provide a brief feedback and a helpful 'Search Hint'.\n"
            "- Mention the Final Result and Total Score."
        )
    else:
        task_instruction = (
            "- Provide a concise, professional technical assessment.\n"
            f"- {scenario_feedback}\n"
            "- List key strengths and weaknesses based on the history.\n"
            "- State the Final Result and Total Score clearly."
        )
    
    prompt = (
        f"Role: {req.session.interviewer_personality} Interviewer. Scenario: {req.session.scenario}.\n"
        f"Candidate: {req.session.name}, Track: {req.session.track_name}.\n"
        f"INTERVIEW HISTORY:\n{history_str}\n"
        f"STATISTICS: Total Score: {total_score}, Status: {is_passed}.\n\n"
        f"TASK: {task_instruction}\n\n"
        f"CRITICAL RULE: Output ONLY the summary text. No labels, no markdown."
    )
    
    try:
        output = client.chat_completion(
            [{"role": "user", "content": prompt}],
            max_tokens=1000
        )

        return {
            "summary": clean_api_response(output.choices[0].message.content),
            "total_score": total_score,
            "status": is_passed,
            "average": round(avg_score, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from typing import List
import re

@app.post("/generate_ideal_answer")
async def generate_ideal_answer(data: IdealAnswerRequest):

    def clean_text(text: str) -> str:
        text = text.replace("Here is the ideal answer:", "")
        text = text.replace("Here’s the ideal answer:", "")

        text = re.sub(r"[.,;:!?()\[\]{}\"']", "", text)

        text = re.sub(r"\s+", " ", text).strip()

        return text

    results = []

    for q in data.questions:

        prompt = (
            f"You are a Senior {data.session.track_name} Architect.\n"
            f"Question: {q}\n\n"
            f"TASK:\n"
            f"- Give a SHORT ideal answer 5 to 8 lines max\n"
            f"- Be technical and direct\n"
            f"- DO NOT add introductions or conclusions\n"
            f"- ONLY key points\n\n"
            f"OUTPUT RULE:\n"
            f"- Return only the answer"
        )

        output = client.chat_completion(
            [{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3
        )

        raw_answer = output.choices[0].message.content.strip()

        cleaned_answer = clean_text(raw_answer)
        cleaned_answer = clean_api_response(cleaned_answer)

        results.append({
            "question": q,
            "ideal_answer": cleaned_answer
        })

    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
