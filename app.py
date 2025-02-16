# app.py
# /// script
# dependencies = [
#   "requests",
#   "fastapi",
#   "uvicorn",
#   "python-dateutil",
#   "pandas",
#   "db-sqlite3",
#   "scipy",
#   "pybase64",
#   "python-dotenv",
#   "httpx",
#   "markdown",
#   "duckdb",
#   "pillow",
#   "pytesseract",
#   "flask",
#   "whisper",
#   "openai",
#   "numpy",
#   "torch",
#   "openai-whisper",
#   "opencv-python",
#   "langdetect",
#   "cryptography"
# ]
# ///
 
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from tasksA import *
from tasksB import *
from dotenv import load_dotenv
import os
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Response, Query
from pydantic import BaseModel
import pandas as pd
from io import StringIO
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


app = FastAPI()
load_dotenv()


@app.get("/ask")
def ask(prompt: str):
    result = get_completions(prompt)
    return result

openai_api_chat_url= "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
openai_api_key = os.getenv("AIPROXY_TOKEN")

headers = {
    "Authorization": f"Bearer {openai_api_key}",
    "Content-Type": "application/json",
}

task_function = [
  {
        "name": "A1",
        "description": "Run a Python script from a given URL, passing an email as the argument.",
        "parameters": {
            "type": "object",
            "properties": {
                 "filename": {"type": "string", "pattern": r"https?://.*\.py"},
                 "targetfile": {"type": "string", "pattern": r".*/(.*\.py)"},
                "email": {"type": "string", "pattern": r"[\w\.-]+@[\w\.-]+\.\w+"}
            },
            "required": ["filename", "targetfile", "email"]
        }
    },
    {
        "name": "A2",
        "description": "Format a markdown file using a specified version of Prettier.",
        "parameters": {
            "type": "object",
            "properties": {
                "prettier_version": {"type": "string", "pattern": r"prettier@\d+\.\d+\.\d+"},
                "filename": {"type": "string", "pattern": r".*/(.*\.md)"}
            },
            "required": ["prettier_version", "filename"]
        }
    },
    {
        "name": "A3",
        "description": "Count the number of occurrences of a specific weekday in a date file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "pattern": r"/data/.*dates.*\.txt"},
                "targetfile": {"type": "string", "pattern": r"/data/.*/(.*\.txt)"},
                "weekday": {"type": "integer", "pattern": r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"}
            },
            "required": ["filename", "targetfile", "weekday"]
        }
    },
    {
        "name": "A4",
        "description": "Sort a JSON contacts file and save the sorted version to a target file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "pattern": r".*/(.*\.json)",
                },
                "targetfile": {
                    "type": "string",
                    "pattern": r".*/(.*\.json)",
                }
            },
            "required": ["filename", "targetfile"]
        }
    },
    {
        "name": "A5",
        "description": "Retrieve the most recent log files from a directory and save their content to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "log_dir_path": {
                    "type": "string",
                    "pattern": r".*/logs",
                    "default": "/data/logs"
                },
                "output_file_path": {
                    "type": "string",
                    "pattern": r".*/(.*\.txt)",
                    "default": "/data/logs-recent.txt"
                },
                "num_files": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 10
                }
            },
            "required": ["log_dir_path", "output_file_path", "num_files"]
        }
    },
    {
        "name": "A6",
        "description": "Generate an index of documents from a directory and save it as a JSON file.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_dir_path": {
                    "type": "string",
                    "pattern": r".*/docs",
                    "default": "/data/docs"
                },
                "output_file_path": {
                    "type": "string",
                    "pattern": r".*/(.*\.json)",
                    "default": "/data/docs/index.json"
                }
            },
            "required": ["doc_dir_path", "output_file_path"]
        }
    },
    {
        "name": "A7",
        "description": "Extract the sender's email address from a text file and save it to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "pattern": r".*/(.*\.txt)",
                    "default": "/data/email.txt"
                },
                "output_file": {
                    "type": "string",
                    "pattern": r".*/(.*\.txt)",
                    "default": "/data/email-sender.txt"
                }
            },
            "required": ["filename", "output_file"]
        }
    },
    {
        "name": "A8",
        "description": "Generate an image representation of credit card details from a text file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "pattern": r".*/(.*\.txt)",
                    "default": "/data/credit-card.txt"
                },
                "image_path": {
                    "type": "string",
                    "pattern": r".*/(.*\.png)",
                    "default": "/data/credit_card.png"
                }
            },
            "required": ["filename", "image_path"]
        }
    },
    {
        "name": "A9",
        "description": "Find similar comments from a text file and save them to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "pattern": r".*/(.*\.txt)",
                    "default": "/data/comments.txt"
                },
                "output_filename": {
                    "type": "string",
                    "pattern": r".*/(.*\.txt)",
                    "default": "/data/comments-similar.txt"
                }
            },
            "required": ["filename", "output_filename"]
        }
    },
    {
        "name": "A10",
        "description": "Identify high-value (gold) ticket sales from a database and save them to a text file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "pattern": r".*/(.*\.db)",
                    "default": "/data/ticket-sales.db"
                },
                "output_filename": {
                    "type": "string",
                    "pattern": r".*/(.*\.txt)",
                    "default": "/data/ticket-sales-gold.txt"
                },
                "query": {
                    "type": "string",
                    "pattern": "SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'"
                }
            },
            "required": ["filename", "output_filename", "query"]
        }
    },
    {
        "name": "B1",
        "description": "Check if filepath starts with /data",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "pattern": r"^/data/.*",
                    # "description": "Filepath must start with /data to ensure secure access."
                }
            },
            "required": ["filepath"]
        }
    },
    {
    "name": "B2",
    "description": "Ensure that no file is deleted from the file system.",
    "parameters": {
        "type": "object",
        "properties": {
            "filepath": {
                "type": "string",
                "pattern": "^/data/.*",
                "description": "Filepath must start with /data to ensure the file is within the secure directory."
            }
        },
        "required": ["filepath"]
    }
    },
 {
    "name": "B3",
    "description": "Download content from an API URL and save it to the specified path.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "pattern": r"https?://.*\.(json|xml|api).*",
                "description": "URL to download content from. Must be an API endpoint."
            },
            "output_file": {
                "type": "string",
                "pattern": r".*/.*",
                "description": "Path to save the downloaded content."
            }
        },
        "required": ["url", "output_file"]
    }
},
    {
    "name": "B4",
    "description": "Clone a Git repository to the /data/repo directory and make a commit with the given message.",
    "parameters": {
        "type": "object",
        "properties": {
            "repo_url": {
                "type": "string",
                "format": "url",
                "description": "The URL of the Git repository to clone."
            },
            "commit_message": {
                "type": "string",
                "description": "The commit message to be used after the repository is cloned."
            }
        },
        "required": ["repo_url", "commit_message"]
    }
},
    {
        "name": "B5",
        "description": "Execute a SQL query on a specified database file and save the result to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "db_path": {
                    "type": "string",
                    "pattern": r".*/(.*\.db)",
                    "description": "Path to the SQLite database file."
                },
                "query": {
                    "type": "string",
                    "description": "SQL query to be executed on the database."
                },
                "output_filename": {
                    "type": "string",
                    "pattern": r".*/(.*\.txt)",
                    "description": "Path to the file where the query result will be saved."
                }
            },
            "required": ["db_path", "query", "output_filename"]
        }
    },
    {
        "name": "B6",
        "description": "Fetch content from a URL and save it to the specified output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "pattern": r"https?://.*",
                    "description": "URL to fetch content from."
                },
                "output_filename": {
                    "type": "string",
                    "pattern": r".*/.*",
                    "description": "Path to the file where the content will be saved."
                }
            },
            "required": ["url", "output_filename"]
        }
    },
    {
        "name": "B7",
        "description": "Process an image by optionally resizing it and saving the result to an output path.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "pattern": r".*/(.*\.(jpg|jpeg|png|gif|bmp))",
                    "description": "Path to the input image file."
                },
                "output_path": {
                    "type": "string",
                    "pattern": r".*/.*",
                    "description": "Path to save the processed image."
                },
                "resize": {
                    "type": "array",
                    "items": {
                        "type": "integer",
                        "minimum": 1
                    },
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Optional. Resize dimensions as [width, height]."
                }
            },
            "required": ["image_path", "output_path"]
        }
    },{
    "name": "B8",
    "description": "Transcribe audio using OpenAI's Whisper model from the given audio file path.",
    "parameters": {
        "type": "object",
        "properties": {
            "audio_path": {
                "type": "string",
                "description": "The file path of the audio file to be transcribed."
            }
        },
        "required": ["audio_path"]
    }
},
    {
        "name": "B9",
        "description": "Convert a Markdown file to another format and save the result to the specified output path.",
        "parameters": {
            "type": "object",
            "properties": {
                "md_path": {
                    "type": "string",
                    "pattern": r".*/(.*\.md)",
                    "description": "Path to the Markdown file to be converted."
                },
                "output_path": {
                    "type": "string",
                    "pattern": r".*/.*",
                    "description": "Path where the converted file will be saved."
                }
            },
            "required": ["md_path", "output_path"]
        }
    },{
    "name": "B10",
    "description": "Write an API endpoint that filters a CSV file and returns JSON data",
    "parameters": {
        "type": "object",
        "properties": {
            "csv_path": {
                "type": "string",
                "description": "Path to the CSV file to be filtered. It must start with '/data'."
            },
            "filter_column": {
                "type": "string",
                "description": "Column in the CSV file to be used for filtering the data."
            },
            "filter_value": {
                "type": "string",
                "description": "The value in the filter column used to filter the data."
            }
        },
        "required": ["csv_path", "filter_column", "filter_value"]
    },
    "responses": {
        "200": {
            "description": "Successfully filtered data",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "array",
                        "items": {
                            "type": "object"
                        }
                    }
                }
            }
        },
        "400": {
            "description": "Invalid input or missing parameter",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error": {
                                "type": "string",
                                "description": "Error message describing what went wrong"
                            }
                        }
                    }
                }
            }
        },
        "404": {
            "description": "CSV file does not exist",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error": {
                                "type": "string",
                                "description": "Error message indicating the CSV file is not found"
                            }
                        }
                    }
                }
            }
        },
        "500": {
            "description": "Server error during processing",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error": {
                                "type": "string",
                                "description": "Error message describing an internal server issue"
                            }
                        }
                    }
                }
            }
        }
    },
    
}, {
        "name": "Bonus1",
        "description": "Summarize text from a file and save the summary.",
        "parameters": {
            "type": "object",
            "properties": {
                "text_path": {"type": "string"},
                "output_filename": {"type": "string"}
            },
            "required": ["text_path", "output_filename"]
        }
    },
    {
        "name": "Bonus2",
        "description": "Convert an image to grayscale and save it.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "output_filename": {"type": "string"}
            },
            "required": ["image_path", "output_filename"]
        }
    },
    {
        "name": "Bonus3",
        "description": "Extract key frames from a video and save as images.",
        "parameters": {
            "type": "object",
            "properties": {
                "video_path": {"type": "string"},
                "output_directory": {"type": "string"}
            },
            "required": ["video_path", "output_directory"]
        }
    },
    {
        "name": "Bonus4",
        "description": "Fetch JSON data from an API and save it.",
        "parameters": {
            "type": "object",
            "properties": {
                "api_endpoint": {"type": "string"},
                "output_filename": {"type": "string"}
            },
            "required": ["api_endpoint", "output_filename"]
        }
    },
    {
        "name": "Bonus5",
        "description": "Encrypt a text file and save the encrypted version.",
        "parameters": {
            "type": "object",
            "properties": {
                "input_filename": {"type": "string"},
                "output_filename": {"type": "string"}
            },
            "required": ["input_filename", "output_filename"]
        }
    },
    {
        "name": "Bonus6",
        "description": "Extract unique values from a CSV column and save them.",
        "parameters": {
            "type": "object",
            "properties": {
                "csv_path": {"type": "string"},
                "column_name": {"type": "string"},
                "output_filename": {"type": "string"}
            },
            "required": ["csv_path", "column_name", "output_filename"]
        }
    },
    {
        "name": "Bonus7",
        "description": "Detect language of a text file and save the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "text_path": {"type": "string"},
                "output_filename": {"type": "string"}
            },
            "required": ["text_path", "output_filename"]
        }
    },
    {
        "name": "Bonus8",
        "description": "Filter error logs from a log file and save them.",
        "parameters": {
            "type": "object",
            "properties": {
                "logs_path": {"type": "string"},
                "output_filename": {"type": "string"}
            },
            "required": ["logs_path", "output_filename"]
        }
    },
    {
        "name": "Bonus9",
        "description": "Format a JSON file to be more readable.",
        "parameters": {
            "type": "object",
            "properties": {
                "json_path": {"type": "string"},
                "output_filename": {"type": "string"}
            },
            "required": ["json_path", "output_filename"]
        }
    },
    {
        "name": "Bonus10",
        "description": "Count occurrences of a word in a text file and save it.",
        "parameters": {
            "type": "object",
            "properties": {
                "text_path": {"type": "string"},
                "word": {"type": "string"},
                "output_filename": {"type": "string"}
            },
            "required": ["text_path", "word", "output_filename"]
        }
    }

]

import os
import httpx

# AI Proxy settings
AIPROXY_API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

# Headers for the AI Proxy request
HEADERS = {
    "Authorization": f"Bearer {AIPROXY_TOKEN}",
    "Content-Type": "application/json",
}

def get_completions(prompt: str):
    """Sends a request to AI Proxy's Chat Completion API and extracts structured parameters."""
    try:
        with httpx.Client(timeout=20) as client:
            response = client.post(
                AIPROXY_API_URL,
                headers=HEADERS,
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a function classifier that extracts structured parameters from queries."},
                        {"role": "user", "content": prompt}
                    ],
                    "tools": [{"type": "function", "function": task} for task in task_function],
                    "tool_choice": "auto"
                },
            )

        # Convert response to JSON
        response_json = response.json()
        print("API Response:", response_json)  # Debugging log

        # Ensure response contains "choices"
        if "choices" not in response_json or not response_json["choices"]:
            raise ValueError(f"Unexpected API response: Missing 'choices' key - {response_json}")

        # Ensure "message" key exists in the first choice
        first_choice = response_json["choices"][0]
        if "message" not in first_choice or "tool_calls" not in first_choice["message"]:
            raise ValueError(f"Invalid API response structure - {response_json}")

        # Ensure "tool_calls" exists and contains at least one function call
        tool_calls = first_choice["message"]["tool_calls"]
        if not tool_calls or "function" not in tool_calls[0]:
            raise ValueError(f"Invalid tool_calls structure - {response_json}")

        return tool_calls[0]["function"]

    except httpx.RequestError as e:
        print(f"HTTP Request failed: {str(e)}")
        return {"error": "API request failed", "details": str(e)}

    except ValueError as e:
        print(f"Response parsing error: {str(e)}")
        return {"error": "Unexpected API response", "details": str(e)}

    except Exception as e:
        print(f"General Error: {str(e)}")
        return {"error": "An unexpected error occurred", "details": str(e)}




@app.post("/run")
async def run_task(task: str):
    try:
        response = get_completions(task)
        print(response)
        task_code = response['name']
        arguments = response['arguments']

        if "A1"== task_code:
            A1(**json.loads(arguments))
        if "A2"== task_code:
            A2(**json.loads(arguments))
        if "A3"== task_code:
            A3(**json.loads(arguments))
        if "A4"== task_code:
            A4(**json.loads(arguments))
        if "A5"== task_code:
            A5(**json.loads(arguments))
        if "A6"== task_code:
            A6(**json.loads(arguments))
        if "A7"== task_code:
            A7(**json.loads(arguments))
        if "A8"== task_code:
            A8(**json.loads(arguments))
        if "A9"== task_code:
            A9(**json.loads(arguments))
        if "A10"== task_code:
            A10(**json.loads(arguments))

    # For phase B execution 
        if "B1"== task_code:
            B1(**json.loads(arguments))
        if "B2"== task_code:
            B2(**json.loads(arguments))
        if "B3" == task_code:
            B3(**json.loads(arguments))
        if "B4"== task_code:
            B4(**json.loads(arguments))
        if "B5" == task_code:
            B5(**json.loads(arguments))
        if "B6" == task_code:
            B6(**json.loads(arguments))
        if "B7" == task_code:
            B7(**json.loads(arguments))
        if "B8"== task_code:
            B8(**json.loads(arguments))
        if "B9" == task_code:
            B9(**json.loads(arguments))
        if task_code == "B10":

            app = FastAPI()

            # Request Model
            class FilterRequest(BaseModel):
                csv_path: str
                filter_column: str
                filter_value: str

            @app.post("/filter_csv")
            async def filter_csv(request: FilterRequest):
                """API endpoint to filter CSV and return JSON data."""
                
                try:
                    result, status_code = B10(request.csv_path, request.filter_column, request.filter_value)

                    # Handle different response statuses
                    if status_code == 400:
                        raise HTTPException(status_code=400, detail=result["error"])
                    elif status_code == 404:
                        raise HTTPException(status_code=404, detail=result["error"])
                    elif status_code == 500:
                        raise HTTPException(status_code=500, detail=result["error"])
                    
                    return result  # Return JSON data

                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
        if "Bonus1" == task_code:
            Bonus1(**json.loads(arguments))
        if "Bonus2" == task_code:
            Bonus2(**json.loads(arguments))
        if "Bonus3" == task_code:
            Bonus3(**json.loads(arguments))
        if "Bonus4" == task_code:
            Bonus4(**json.loads(arguments))
        if "Bonus5" == task_code:
            Bonus5(**json.loads(arguments))
        if "Bonus6" == task_code:
            Bonus6(**json.loads(arguments))
        if "Bonus7" == task_code:
            Bonus7(**json.loads(arguments))
        if "Bonus8" == task_code:
            Bonus8(**json.loads(arguments))
        if "Bonus9" == task_code:
            Bonus9(**json.loads(arguments))
        if "Bonus10" == task_code:
            Bonus10(**json.loads(arguments))


        return {"message": f"{task_code} Task '{task}' executed successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str = Query(..., description="File path to read")):
    try:
        with open(path, "r") as file:
            return file.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)





