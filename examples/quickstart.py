"""New Python example: independently compute one scattering point."""
from puffy.physics import solve, born_S
import json
if __name__ == "__main__":
    result = solve(b=2, kappa=.35, ychi=1., yn=.5)
    result["born_S"] = born_S(2, .35, 1., .5)
    print(json.dumps(result, indent=2))
