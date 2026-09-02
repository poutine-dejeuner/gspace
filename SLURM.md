# Règles Slurm — Cluster `cursor`

les compute nodes utilisent des gpu b300. Nous avons eut des difficultés
a construire des venv qui les supportent correctement, considère ~/vllm-env
pour voir d'ou commencer quand tu créé un nouveau venv.

## Toujours demander le **minimum** de ressources

Chaque job slurm **doit** spécifier explicitement CPUs, RAM, et GPUs. **Ne jamais** laisser slurm allouer
par défaut — sinon il prend le nœud entier.

```bash
sbatch \
    --gres=gpu:1 \           # 1 GPU (jamais plus sans raison)
--cpus-per-task=8 \      # 4–16 selon le besoin réel
--mem=32G \              # 16G–128G selon le besoin réel
--time=4:00:00 \         # toujours mettre une limite
--job-name=... \
    script.sh
```

Avant de soumettre, vérifier ce qui tourne déjà :

```bash
ssh cursor squeue -u \$USER
```

## Règles

1. **--mem, --cpus-per-task, --gres sont obligatoires** sur chaque `sbatch` / `srun`.
2. **1 GPU max** par job, sauf si le code fait du multi-GPU explicite (DDP, FSDP).
3. **Pas de `--exclusive`**.
4. **Pas de job qui monopolise plusieurs nœuds** sans validation explicite.
5. Si tous les nœuds sont occupés, **attendre** — ne pas forcer avec `--qos` ou `--partition` alternatif.
