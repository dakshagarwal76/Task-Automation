# 🤖 LLM-Based Automation Agent

This project is an intelligent automation agent that interprets natural language task instructions and performs the requested actions using a combination of deterministic programming and Large Language Model (LLM)-based reasoning. The agent is built to simplify routine tasks by allowing users to describe what they want in plain English.

---

## 🌐 API Endpoints

### `POST /run?task=<task description>`

Executes a task described in natural language. Internally, the agent parses the instruction using GPT-4o-Mini, determines the required operations, and executes the task. Returns:

- `200 OK` on success  
- `400 Bad Request` for invalid instructions  
- `500 Internal Server Error` for execution failures

### `GET /read?path=<file path>`

Returns the content of a specified file within the `/data` directory as plain text. Used to verify the output of tasks.

---

## ✅ Supported Tasks

The agent supports a range of structured and unstructured tasks, grouped into **Task Set A (Operations)**, **Task Set B (Business Use Cases)**, and a set of additional **Bonus Tasks**.

---

## 🛠 Task Set A: Operational Tasks

- **A1**: Run a Python script from a remote URL with user-specific input (e.g., an email address).
- **A2**: Format a Markdown file in-place using a specified version of `prettier`.
- **A3**: Count the number of specific weekdays (e.g., Wednesdays) in a text file of dates and write the count to another file.
- **A4**: Sort a list of contacts in a JSON file by `last_name` and `first_name`, and save the result.
- **A5**: From the 10 most recent `.log` files, write the first line of each (ordered by recency) to a summary file.
- **A6**: Extract the first `#` (H1) title from all Markdown files in a directory and generate an index mapping filenames to titles.
- **A7**: Extract the sender’s email address from a plain-text email using LLM and write it to a file.
- **A8**: Extract a credit card number from an image file using LLM and write it without spaces to a file.
- **A9**: Use semantic embeddings to find the most similar pair of comments from a list and write them line-by-line to a file.
- **A10**: Query a local SQLite database and calculate the total sales from all “Gold” ticket type entries.

---

## 📈 Task Set B: Business Use Cases

- **B1**: Ensure that no access occurs outside the `/data` directory.
- **B2**: Prevent any deletion of files or directories.
- **B3**: Fetch data from a remote API and save it locally.
- **B4**: Clone a Git repository, make changes, and commit them.
- **B5**: Run a SQL query on SQLite or DuckDB databases and store results.
- **B6**: Scrape data from a website and write structured output.
- **B7**: Compress or resize an image and save the result.
- **B8**: Transcribe audio from an `.mp3` file to text.
- **B9**: Convert Markdown files to HTML.
- **B10**: Filter a CSV file based on specific criteria and return the result as JSON through an API.

---

## 🏆 Bonus Tasks

The agent also supports advanced or creative tasks that demonstrate flexible automation capabilities:

- **Bonus1**: Summarize a text file (first 200 characters as a placeholder summary) and save it.
- **Bonus2**: Convert an image to grayscale and save it.
- **Bonus3**: Extract key frames from a video (every 30th frame) and save them as images.
- **Bonus4**: Fetch JSON data from an external API and save it in formatted form.
- **Bonus5**: Encrypt the contents of a text file using a generated key and save the encrypted file.
- **Bonus6**: Extract and save all unique values from a specified column in a CSV file.
- **Bonus7**: Detect the language of a given text file and write the result to a file.
- **Bonus8**: Filter out only error lines (containing “ERROR”) from a log file and save them.
- **Bonus9**: Pretty-print (format) a JSON file with indentation and write it back to disk.
- **Bonus10**: Count how many times a specific word appears in a text file and write the result.

---

## 🔐 Security & Restrictions

To maintain safe operation and predictable behavior:

- All file operations are **restricted to the `/data` directory**
- **No delete operations** are permitted
- Output is always **deterministic and verifiable**
- Each task is designed to complete within **20 seconds**

---

## 📋 Example Usage

```bash
# Submit a task to count Wednesdays
curl -X POST "http://localhost:8000/run?task=Count+Wednesdays+in+/data/dates.txt+and+save+to+/data/output.txt"

# Read the result
curl "http://localhost:8000/read?path=/data/output.txt"
