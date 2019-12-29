FROM kaggle/python-gpu-build:latest

# set dotfiles
WORKDIR /root
RUN git clone https://github.com/guchio3/guchio_utils.git
RUN rm .bashrc && ln -s /root/guchio_utils/.bashrc .bashrc

# set jupyter notebook
# jupyter vim key-bind settings
RUN pip install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN mkdir -p $(jupyter --data-dir)/nbextensions
RUN git clone https://github.com/lambdalisue/jupyter-vim-binding $(jupyter --data-dir)/nbextensions/vim_binding
RUN jupyter nbextension enable vim_binding/vim_binding
# edit vim_bindings setting as I can use C-c for exitting insert mode
RUN sed -i "s/      'Ctrl-C': false,  \/\/ To enable clipboard copy/\/\/      'Ctrl-C': false,  \/\/ To enable clipboard copy/g" $(jupyter --data-dir)/nbextensions/vim_binding/vim_binding.js
