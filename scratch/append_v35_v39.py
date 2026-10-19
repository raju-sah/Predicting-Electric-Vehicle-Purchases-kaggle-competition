rows = [
    "v35_zoom_max_pass1,Merged Zoom-Zoom-Max solo LGBM (TE+windows+16384-bin income stats+modifiers),0.945912,,solo below 0.94649 anchor; pass2 skipped per gate; fold state in scratch/v35_state",
    "v36_apex_upgraded_slots,Apex Upgraded Slots (90% Pure49 lexrank + 6% SixView+RealMLP3+XGB10f + 4% RealMLP3 + lexsort),,0.94650,Ties best; upgraded Sep-16 stack slots hold 0.94650 at 10% injection; replicates megayak published recipe",
    "v38_apex_hillclimb12,Apex Hillclimb 12 (88% Pure49 + 12% hill-climbed Sep14 stack + boundary + lexsort),,0.94650,Ties best at 12% injection; new stack stronger than v34 8% old stack which dropped to 0.94648",
    "v39_apex_probit_power12,Apex Probit Power 12 (88% Pure49 probit + 12% 0.74-power stack injection),,0.94649,-0.00001 vs rank-space; probit transform of injection loses a digit; rank-space preferred",
    "v37_megayak_stack_probe,Megayak OOF-stack rebuild as standalone second anchor probe,,NOT submitted,Superseded by v37b hill-climb; standalone stack would read ~0.94644 per published recipe",
]
with open("exp_log.csv", "a") as f:
    for r in rows:
        f.write(r + "\n")
print("appended", len(rows), "rows")
