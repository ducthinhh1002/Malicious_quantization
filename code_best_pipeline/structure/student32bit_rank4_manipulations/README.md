# Same seven manipulated teacher datasets, student LoRA rank4

Teacher is unchanged: original Structure32 teacher, rank32, refined600steps, adapter SHA256 `2f9fc8d664786acfe50bf7d47c20e037a92e4cc61805524233bd7f15f365a2d8`. Teacher is not retrained and the later robust1200 teacher is not used.

Seven independent clean SD1.4 students train fresh **rank4 attention+convolution LoRA** to20000steps, batch4, seed0,5000teacher images each, preserving the rank32 experiment's source pixel mapping and fixed transformations: flip; uniformrotation±15° nearest grayfill/RGB8; randomsquarecrop75%area bilinear tensor; keep96%pixels/restgray tensor; Gaussianσ0.05 clip tensor; brightness0.7 RGB8; blurkernel15 σUniform[0.1,2] RGB8.

Pinned trainer, LR schedule (1e-4 through7500, cosine to2e-5 by12500, constant2e-5 afterward), deterministic epoch sampler, RNG-preserving image augmentation, and evaluation remain the same. Full optimizer/scaler/RNG checkpoints every1250steps. There is no12500interruption/evaluation in this run; full-state12500 remains available. Each branch passes a four-update real-model smoke before training. AMP attempted/applied/skipped counts are logged and reported separately; more than1 skipped step stops evaluation for inspection. No old student weights are used.

Evaluate1000 unmanipulated native student outputs at fixed20000. Save BA, exact32/32 FK, acceptance≥24/32 for group1/8/32/64/256 plus image utility and pairedLPIPS. Compare against the matching rank32 branch, not against a different manipulation. Single seed and identical hyperparameters compare these concrete configurations; they do not guarantee each rank has its individually optimal LR or bitwise-equivalent GPU computation.

Remote root `/var/lib/structure32_student_manipulations_rank4`. Four concurrent slots (two per GPU on.21), three remaining branches queued. Source code and frozen identities are copied back under deployed/. Existing rank32 runs/checkpoints are preserved.
