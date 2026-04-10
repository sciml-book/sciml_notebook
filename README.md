# Scientific Machine Learning: Foundations, Methods, and Applications

> Krishna Kumar

"Scientific Machine Learning: Foundations, Methods, and Applications" uniquely bridges traditional scientific computing and cutting-edge machine learning techniques. This comprehensive guide addresses a critical gap in Scientific Machine Learning (SciML) by providing rigorous mathematical foundations alongside practical implementations.

While SciML offers powerful tools for modeling complex systems, it often lacks the mathematical guarantees that traditional numerical methods provide. This book tackles this issue head-on, extending fundamental theorems to ensure guarantees for both functions and their derivatives—crucial for solving partial differential equations. These theoretical advances have direct practical implications, informing the choice of activation functions in Physics-Informed Neural Networks and translating into better neural network design, error bounds, and convergence rates.

The book covers a wide range of topics, from physics-informed neural networks and neural operators to differentiable programming and generative models for scientific applications. It demonstrates how mathematical insights elevate SciML beyond mere hyperparameter tuning to principled, mathematically grounded modeling.

With clear explanations, real-world case studies, and code examples in popular frameworks like PyTorch and JAX, this book is indispensable for anyone seeking to fully exploit AI's potential in scientific and engineering applications while maintaining mathematical rigor. Whether a graduate student, researcher, or industry professional, you'll gain the knowledge and tools to design more accurate, interpretable, and physically consistent models.

## Environment setup

The notebook environment is defined in [pyproject.toml](/Users/krishna/courses/CE397-Scientific-MachineLearning/book/sciml_notebook/pyproject.toml).
Use `uv` to create the virtual environment and install the notebook stack.

```bash
cd sciml_notebook
uv venv
uv sync
```

Add `--extra docs` when you want the MkDocs toolchain.

```bash
uv sync --extra docs
```

The root `requirements.txt` mirrors the runtime notebook dependencies for users who prefer `pip`.
The file `requirements-docs.txt` contains the documentation stack.
