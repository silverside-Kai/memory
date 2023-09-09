if __name__ == "__main__":
    import uvicorn

    uvicorn.run("modules.api_chatbot:app",
                host="0.0.0.0", reload=True, port=8080)
