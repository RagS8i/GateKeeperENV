Files placed here are mounted read-only into the harness container when it starts, so
changing them needs no rebuild. They are never visible to the agent's sandbox.

To add extra instructions to the system prompt, put a text file here and name it in `.env`:

    APPEND_FILE=inputs/extra.txt

Its contents are added to the end of the system prompt exactly as written. If `APPEND_FILE`
is empty, `append.txt` in this folder is used when it exists. A named file that is missing,
empty or not UTF-8 stops the run with exit code 3 before anything else happens.

Everything in this folder except this README and `.gitkeep` is ignored by git.
