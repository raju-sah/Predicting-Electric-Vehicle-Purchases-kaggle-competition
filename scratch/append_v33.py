with open("exp_log.csv", "a") as f:
    f.write("v33_probit_latent_apex,Latent Probit-Space Apex (82% Pure 49 + 8% Taeyang 50 + 4% Deotte + 3% Stacker + 3% RealMLP in Probit Space),0.94650,0.94649,Smooth non-linear probit transformation confirms mathematical alignment with underlying latent generator; holds 0.94649\n")
print("Appended v33 to exp_log.csv")
