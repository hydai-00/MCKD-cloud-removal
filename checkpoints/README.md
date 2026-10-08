# Checkpoints

Two files belong here. Both are epoch 14 of their 15-epoch run, and both are plain
`{'epoch': int, 'state_dict': OrderedDict}` dicts read with `torch.load(path)["state_dict"]`.

| file | size | architecture |
|---|---:|---|
| `teacher.pkl` | 96.5 MB | `model.mckd.Net(13, 2)` |
| `student.pkl` | 97.8 MB | `model.mckd.Net(13, 2, iss=True)` |

Place them, relative to the repository root, at (the scripts read `last.pkl`):

```
main/result/all_My_My_end_KD_teacher/last.pkl
main/result/all_My_My_end_KD_student/last.pkl
```

Then evaluate with the command in README section 5, or re-run stage 2 alone with
`--student_only True --teacher_path ./result/all_My_My_end_KD_teacher/last.pkl`.

## Loading them

Both were saved from `nn.DataParallel`, so every key carries a `module.` prefix. The
shipped scripts wrap the network the same way before loading, so the prefix resolves there.
Into an **unwrapped** network, strip exactly one leading `module.`, and keep `strict=True`
so a mismatch stays loud:

```python
sd = torch.load(path, map_location="cpu")["state_dict"]
sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
Net(13, 2, iss=True).load_state_dict(sd, strict=True)
```

`str.replace("module.", "")` is **not** equivalent — the network has layers of its own named
`bottle_neck_module.0`, and a global replace mangles them. Nor is `strict=False` a safe
shortcut: an unwrapped network against a prefixed `state_dict` reports every key as both
missing and unexpected, loads nothing, and leaves the model at its random initialisation
without raising.

## Publishing them

Each file is just under GitHub's 100 MB per-file limit, so a commit is possible but a bad
idea — the warning threshold is 50 MB, and the files would bloat the history permanently.
Attach them to a **release** instead (2 GB per file, not counted against repository size or
Git LFS quota). `*.pkl` is gitignored here, so `git add .` will not pick them up; if you do
want them in the tree, use `git add -f checkpoints/*.pkl`.
