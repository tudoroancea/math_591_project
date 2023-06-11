import os
import numpy as np
import sys
import pandas as pd

# data = np.loadtxt(sys.argv[1], delimiter=",", skiprows=1)
# data = data[::5, :]
# for i in range(0, data.shape[0]-40, )
# np.savetxt(
#     sys.argv[2],
#     data,
#     delimiter=",",
#     header="timestamp,X,Y,phi,v_x,v_y,r,T,delta"
#     if data.shape[1] == 9
#     else "timestamp,X,Y,phi,v_x,T,delta",
# )

for file in os.listdir("data/sysid/dimanche"):
    if file.endswith(".csv"):
        df = pd.read_csv("data/sysid/dimanche/" + file)
        # remove cols starting with rho or theta
        df = df.loc[:, ~df.columns.str.startswith("rho")]
        df = df.loc[:, ~df.columns.str.startswith("theta")]
        # df = df.loc[:, ~df.columns.str.startswith("v_y")]
        # df = df.loc[:, ~df.columns.str.startswith("r")]
        # phi_0 = df["phi"].iloc[0]
        # df["phi"] = df["phi"] - phi_0
        # XY = df[["X", "Y"]].to_numpy()
        # XY_0 = XY[0]
        # angle = np.pi/2 - phi_0
        # XY = (
        #     (XY - XY_0)
        #     @ np.array(
        #         [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        #     ).T
        # )
        # # XY = XY - XY_0
        # df[["X", "Y"]] = XY
        df.to_csv(
            "data/sysid/portes_ouvertes_dyn6/dimanche" + file.removesuffix(".csv"),
            index=False,
        )
