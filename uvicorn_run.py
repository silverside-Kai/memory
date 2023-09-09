if __name__ == "__main__":
    import uvicorn

    uvicorn.run("modules.api_chatbot:app",
                host='localhost', reload=True, port=8080)
