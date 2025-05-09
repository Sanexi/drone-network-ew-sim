Swarm Formation Geometry

The swarm formation is designed to be square, symmetric, and scalable, with specific link lengths being equal. The parameter `LINK_LENGTH_METERS` (let's call it $L_{scale}$ in formulas) acts as the primary scalar and defines the length of Leader-Relay, Relay-Sensor, and Relay-Attack links.

**A. Unit Coordinates Derivation Highlights:**

The positions are first determined in a "unit coordinate" system where the primary link length (M-R, R-S, R-A) is 1 unit. This involves solving a system of geometric equations based on:
1.  **Equal Link Lengths:**
    $d(M_1, R_i) = 1$ (unit)
    $d(R_i, S_j) = 1$ (unit)
    $d(R_i, A_k) = 1$ (unit)
2.  **Square Swarm Footprint:** Max X-extent = Max Y-extent.
    If $S_{x,max}$ is the x-coordinate of the front sensor row and $S_{y,outer}$ is the y-coordinate of the outermost sensor: $S_{x,max} = S_{y,outer}$.
3.  **Even Sensor Spacing:** The Y-coordinates of the four sensors in the front row $(S_1, S_2, S_3, S_4)$ are evenly spaced.

This leads to defining a constant $k = \frac{1}{\sqrt{5}}$. The unit coordinates for each drone relative to the Leader ($M_1$) at $(0,0)$ are then:

* **Leader ($M_1$):** $(0, 0)$
* **Front Relays ($R_1$: F-L, $R_2$: F-R):**
    * $R_1: (k \cdot 1, k \cdot 2)$
    * $R_2: (k \cdot 1, k \cdot -2)$
* **Sensors ($S_1, S_2$ on $R_1$; $S_3, S_4$ on $R_2$):**
    * $S_1: (k \cdot 3, k \cdot 3)$ (Outer-Left)
    * $S_2: (k \cdot 3, k \cdot 1)$ (Inner-Left)
    * $S_3: (k \cdot 3, k \cdot -1)$ (Inner-Right)
    * $S_4: (k \cdot 3, k \cdot -3)$ (Outer-Right)
* **Back Relays ($R_3$: B-L, $R_4$: B-R):**
    * $R_3: (k \cdot -1, k \cdot 2)$
    * $R_4: (k \cdot -1, k \cdot -2)$
* **Attack Drones ($A_1, A_2$ on $R_3$; $A_3, A_4$ on $R_4$):**
    * $A_1: (k \cdot -3, k \cdot 3)$ (Outer-Left)
    * $A_2: (k \cdot -3, k \cdot 1)$ (Inner-Left)
    * $A_3: (k \cdot -3, k \cdot -1)$ (Inner-Right)
    * $A_4: (k \cdot -3, k \cdot -3)$ (Outer-Right)

**B. Scaling to Actual Positions:**
The actual initial relative position of any drone $(x_{rel}, y_{rel})$ with respect to the Leader ($M_1$) is:
$x_{rel} = \text{unit\_coord}_x \times L_{scale}$
$y_{rel} = \text{unit\_coord}_y \times L_{scale}$

The absolute initial position $(x_{abs}, y_{abs})$ is:
$x_{abs} = x_{M1,start} + x_{rel}$
$y_{abs} = y_{M1,start} + y_{rel}$
(Assuming swarm's local X-forward aligns with global X, and local Y-left aligns with global Y at start).
