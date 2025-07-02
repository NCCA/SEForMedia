echo "Loading zsh functions..."
jupyter-lab(){
    uv run --with jupyter jupyter lab
}

jupyter-qtconsole(){
    uv run --with jupyter jupyter qtconsole
}

jupyter-notebook(){
    uv run --with jupyter jupyter notebook
}
