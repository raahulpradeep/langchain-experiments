import os
import logging
from langchain.llms import OpenAI
from langchain.agents import create_csv_agent
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from schemas import ChatResponse

app = FastAPI()
templates = Jinja2Templates(directory="templates")
llm = OpenAI(temperature=0)
sheet_qa_agent = None

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/load/{file}")
async def load(file: str):
    global sheet_qa_agent

    try:
        sheet_qa_agent = create_csv_agent(llm, file, verbose=True)

    except Exception as e:
        logging.error(e)
        return "Error creating agent for " + file + ". Please look at logs for further debugging"
    return "Successfully loaded " + file


@app.websocket("/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            if sheet_qa_agent is None:
                raise Exception("No csv file loaded")

            result = sheet_qa_agent.run(question)
            answer_resp = ChatResponse(sender="bot", message=result, type="stream")
            await websocket.send_json(answer_resp.dict())

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())

        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9001)
