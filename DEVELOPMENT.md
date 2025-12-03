# DEVELOPMENT

## Installing Dependencies

We use [uv](https://docs.astral.sh/uv/) to manage all the python dependencies.

Each top-level directory corresponds to a sub-project, and accordingly there is a dependency group specified for each sub-project.

This means, that if you want to install the dependencies for the `benchmark`, you can run

```sh
uv sync --group benchmark
```

and similarly for the `app` sub-project

```sh
uv sync --group app
```

To install all dependencies across all sub-projects, you can run

```sh
uv sync --all-groups
```
