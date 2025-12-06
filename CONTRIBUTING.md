# Contributing

This document outlines the development guidelines and conventions for this repository. These rules ensure consistency, maintainability, and collaboration across developers and AI agents.

## Code Style

Before committing, ensure your code follows the project style.

Format code:

```bash
cargo fmt
```

Run linter with strict settings:

```bash
cargo clippy --all -- -W clippy::all -W clippy::pedantic -D warnings
```

## Testing

Run all tests before submitting changes:

```bash
cargo test
```

## Benchmarks

This project uses [Criterion](https://bheisler.github.io/criterion.rs/book/) for benchmarking Benchmarks help track performance regressions.

### Running benchmarks

Run all kernel benchmarks:

```bash
cargo bench --bench kernel --features unstable-kernels
```

Run specific kernel benchmark:

```bash
cargo bench --bench kernel --features unstable-kernels -- kernel/gemm
```

### Comparing performance

Save baseline before making changes (**run it at least twice** to ensure GPU warmup for more stable results):

```bash
cargo bench --bench kernel --features unstable-kernels -- --noplot --quiet --save-baseline main
```

Compare current performance against baseline:

```bash
cargo bench --bench kernel --features unstable-kernels -- --noplot --baseline main
```

Compare specific kernel:

```bash
cargo bench --bench kernel --features unstable-kernels -- kernel/gemm --noplot --baseline main
```

## Commit Message Guidelines

This project has a rule on how git commit messages can be formatted. It uses the simplified [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. This leads to messages that are more readable and easy to follow when looking through the project history.

### Commit Message Format

Each commit message is one-line, and consists of mandatory a **type**, a **scope** and a **subject**:

```
<type>(<scope>): <subject>
```

Any line of the commit message cannot be longer 100 characters.

Examples:

```
docs(common): add contributing guidelines file
```

```
build(common): remove unused dependencies
```

```
chore(common): prepare release v0.1.1
```

### Revert

If the commit reverts a previous commit, it should begin with `revert:`, followed by the subject of the reverted commit.

### Type

Must be one of the following:

* **build**: Changes that affect the build system or external dependencies
* **chore**: Maintenance tasks
* **docs**: Documentation only changes
* **feat**: A new feature
* **fix**: A bug fix
* **perf**: A code change that improves performance
* **refactor**: A code change that neither fixes a bug nor adds a feature
* **style**: Changes that do not affect the meaning of the code
* **test**: Adding missing tests or correcting existing tests

### Scope

The scope should be the name of the package affected (as perceived by the person reading the changelog generated from commit messages).

### Subject

The subject contains a brief description of the change:

* use the imperative, present tense: "change" not "changed" nor "changes"
* use lowercase
* no dot (.) at the end
