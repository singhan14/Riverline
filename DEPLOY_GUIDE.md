# 🚀 How to Switch to PostgreSQL (Production)

You are currently using **SQLite (Local)**. To switch to **PostgreSQL (Cloud)** and stop using the local file, follow these steps.

## Step 1: Get a Cloud Database (Free)
1.  Go to **[Neon.tech](https://neon.tech)** (easiest) or **Supabase**.
2.  Sign up and create a new project named `riverline-db`.
3.  Copy the **Connection String** (It looks like: `postgres://user:pass@ep-xyz.aws.neon.tech/neondb?sslmode=require`).

## Step 2: Configure Local Environment
1.  Open your `.env` file in VS Code.
2.  Add a new line:
    ```env
    DATABASE_URL="your-connection-string-from-step-1"
    ```
3.  Restart your Streamlit app (`CTRL+C`, then `streamlit run app.py`).
4.  **Verification**: Look at your terminal. It should now say:
    > `🌍 Using Cloud Database (PostgreSQL)...`

## Step 3: Configure Cloud Deployment (Streamlit Cloud)
1.  Push your code to GitHub.
2.  Deploy on **Streamlit Cloud**.
3.  Go to **App Settings** -> **Secrets**.
4.  Add the same secret:
    ```toml
    DATABASE_URL = "your-connection-string-from-step-1"
    ```

## Step 4: Remove Local SQLite (Optional)
If you never want to use SQLite again:
1.  Delete `memory.sqlite` from your folder.
2.  Open `agent_graph.py`.
3.  Find `build_graph()` and remove the `else:` block that falls back to SQLite.
4.  Change it to:
    ```python
    if not db_url:
        raise ValueError("DATABASE_URL is missing! Production requires Postgres.")
    ```
