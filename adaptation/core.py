"""
core.py - La matematica dell'adattamento anisotropo (solo numpy).

Qui non si leggono ne' scrivono file: si lavora solo con array numpy.

Come sono fatti i dati:
    points     array (n, 2)      coordinate x, y degli n nodi
    triangles  array (m, 3)      per ogni triangolo, i 3 indici dei suoi nodi
    H, M       array (n, 2, 2)   una matrice 2x2 simmetrica per ogni nodo

I passi, nell'ordine in cui si usano:
    1. hessian_lsq          Hessiano del campo (es. pressione) in ogni nodo
    2. abs_hessian          |H|: autovalori presi in valore assoluto
    3. metric_lp_unscaled   metrica "grezza" M0 (manca ancora un fattore D)
    4. clip_metric          limiti su taglia minima/massima e aspect ratio
    5. metric_complexity    complessita' della metrica (~ meta' dei triangoli)
    6. normalize_metric     trova D in modo che la complessita' sia quella voluta
"""
import numpy as np


# =============================================================================
# Piccole funzioni di aiuto
# =============================================================================

def triangle_areas(points, triangles):
    """Area di ogni triangolo (array di m valori)."""
    A = points[triangles[:, 0]]          # primo vertice di ogni triangolo
    B = points[triangles[:, 1]]          # secondo vertice
    C = points[triangles[:, 2]]          # terzo vertice
    # area = 1/2 |prodotto vettoriale (B-A) x (C-A)|
    cross = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - \
        (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
    return 0.5 * np.abs(cross)


def node_neighbors(n_points, triangles):
    """Per ogni nodo, l'insieme (set) dei nodi collegati da un lato."""
    neighbors = []
    for i in range(n_points):
        neighbors.append(set())
    for a, b, c in triangles:
        a, b, c = int(a), int(b), int(c)
        neighbors[a].update([b, c])
        neighbors[b].update([a, c])
        neighbors[c].update([a, b])
    return neighbors


def from_eig(eigenvalues, eigenvectors):
    """Ricostruisce le matrici dai loro autovalori e autovettori.

    Per una matrice 2x2 simmetrica con autovalori l1, l2 e autovettori v1, v2:
        M = l1 * (v1 v1^T) + l2 * (v2 v2^T)
    Qui lo facciamo per tutti i nodi insieme.
    """
    v1 = eigenvectors[:, :, 0]                    # (n, 2) primo autovettore
    v2 = eigenvectors[:, :, 1]                    # (n, 2) secondo autovettore
    l1 = eigenvalues[:, 0]
    l2 = eigenvalues[:, 1]
    v1v1 = v1[:, :, None] * v1[:, None, :]        # (n, 2, 2) matrice v1 v1^T
    v2v2 = v2[:, :, None] * v2[:, None, :]        # (n, 2, 2) matrice v2 v2^T
    return l1[:, None, None] * v1v1 + l2[:, None, None] * v2v2


# =============================================================================
# 1. Hessiano con i minimi quadrati
# =============================================================================

def hessian_lsq(points, values, neighbors, max_condition=1e10):
    """Hessiano del campo `values` in ogni nodo.

    Per ogni nodo i prendiamo i vicini e i vicini dei vicini (due anelli) e
    scriviamo lo sviluppo di Taylor al secondo ordine:

        p_j - p_i = gx*dx + gy*dy + 1/2*Hxx*dx^2 + Hxy*dx*dy + 1/2*Hyy*dy^2

    Una equazione per ogni vicino j, 5 incognite: lo risolviamo ai minimi quadrati.

    Restituisce:
        H          (n, 2, 2) Hessiano; NaN nei nodi dove il calcolo non e' affidabile
        condition  (n,) numero di condizionamento del sistema di ogni nodo
    """
    points = np.asarray(points, dtype=float)[:, :2]
    values = np.asarray(values, dtype=float)
    n = len(points)

    H = np.full((n, 2, 2), np.nan)        # NaN = "non calcolato"
    condition = np.full(n, np.nan)

    for i in range(n):
        # --- vicini su due anelli (senza il nodo i stesso) -------------------
        stencil = set(neighbors[i])
        for j in neighbors[i]:
            stencil.update(neighbors[j])
        stencil.discard(i)

        # servono almeno 5 equazioni per 5 incognite
        if len(stencil) < 5:
            continue

        # --- distanze dal nodo i, rese adimensionali -------------------------
        ids = np.array(sorted(stencil))
        dx = points[ids, 0] - points[i, 0]
        dy = points[ids, 1] - points[i, 1]
        # distanza del vicino piu' lontano
        scale = np.max(np.sqrt(dx**2 + dy**2))
        dx = dx / scale                          # ora dx, dy sono tra -1 e 1:
        dy = dy / scale                          # il sistema e' ben condizionato

        # --- sistema ai minimi quadrati A * beta = b -------------------------
        A = np.column_stack([dx, dy, 0.5 * dx**2, dx * dy, 0.5 * dy**2])
        b = values[ids] - values[i]
        beta, _, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # --- controlli di affidabilita' --------------------------------------
        if rank < 5 or s[-1] <= 0:
            continue
        condition[i] = s[0] / s[-1]
        if condition[i] > max_condition:
            continue

        # beta = [gx, gy, Hxx, Hxy, Hyy] nelle coordinate adimensionali:
        # per tornare a quelle vere le derivate seconde vanno divise per scale^2
        Hxx = beta[2] / scale**2
        Hxy = beta[3] / scale**2
        Hyy = beta[4] / scale**2
        H[i] = [[Hxx, Hxy],
                [Hxy, Hyy]]

    return H, condition


# =============================================================================
# 2. Valore assoluto dell'Hessiano
# =============================================================================

def abs_hessian(H, eigen_floor=1e-8):
    """|H|: stessa matrice ma con autovalori |lambda|, mai sotto eigen_floor.

    Dove H non e' stato calcolato (NaN) si usa H = 0, cioe' "campo piatto".
    """
    H = np.array(H, dtype=float)
    not_valid = ~np.isfinite(H).all(axis=(1, 2))
    H[not_valid] = 0.0

    eigenvalues, eigenvectors = np.linalg.eigh(H)
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = np.maximum(eigenvalues, eigen_floor)
    return from_eig(eigenvalues, eigenvectors)


# =============================================================================
# 3. Metrica Lp (senza il fattore D)
# =============================================================================

def metric_lp_unscaled(Habs, p=2.0):
    """Metrica che minimizza l'errore di interpolazione in norma Lp (in 2D):

        M0 = det(|H|) ^ (-1 / (2p + 2)) * |H|

    La metrica vera e' M = D * M0; D si calcola con normalize_metric.
    """
    if p < 1:
        raise ValueError("p deve essere >= 1")
    det = np.linalg.det(Habs)                      # un determinante per nodo
    factor = det ** (-1.0 / (2.0 * p + 2.0))
    return factor[:, None, None] * Habs


# =============================================================================
# 4. Limiti sulla metrica
# =============================================================================

def clip_metric(M, h_min=None, h_max=None, max_aspect_ratio=None):
    """Applica i limiti: h_min <= taglia <= h_max e allungamento <= max_aspect_ratio.

    Un autovalore mu della metrica corrisponde alla taglia h = 1 / sqrt(mu), quindi:
        taglia <= h_max   <=>   mu >= 1 / h_max^2
        taglia >= h_min   <=>   mu <= 1 / h_min^2
    None = nessun limite.
    """
    if h_min is None and h_max is None and max_aspect_ratio is None:
        return np.array(M, dtype=float)

    # mu[:, 0] e' il piu' piccolo, mu[:, 1] il piu' grande
    mu, V = np.linalg.eigh(M)

    if h_max is not None:
        mu = np.maximum(mu, 1.0 / h_max**2)
    if h_min is not None:
        mu = np.minimum(mu, 1.0 / h_min**2)

    if max_aspect_ratio is not None:
        # h_lungo / h_corto = sqrt(mu_grande / mu_piccolo) <= max_aspect_ratio
        # Se e' troppo, alziamo mu_piccolo (accorciamo il lato lungo):
        # cosi' la direzione fine, quella che serve sull'urto, non cambia.
        min_allowed = mu[:, 1] / max_aspect_ratio**2
        mu[:, 0] = np.maximum(mu[:, 0], min_allowed)

    return from_eig(mu, V)


# =============================================================================
# 5. Complessita'
# =============================================================================

def metric_complexity(M, points, triangles):
    """C = integrale di sqrt(det M) sul dominio.

    Numero di triangoli della mesh che Gmsh generera' ~ 2 * C.
    Integrale: per ogni triangolo, area * media di sqrt(det M) sui 3 vertici.
    """
    density = np.sqrt(np.linalg.det(M))                 # un valore per nodo
    mean_on_triangle = density[triangles].mean(axis=1)  # media sui 3 vertici
    return float(np.sum(triangle_areas(points, triangles) * mean_on_triangle))


# =============================================================================
# 6. Normalizzazione: scelta di D
# =============================================================================

def normalize_metric(M0, points, triangles, target, h_min=None, h_max=None,
                     max_aspect_ratio=None, tol=1e-4):
    """Cerca D tale che la metrica finale  M = clip(D * M0)  abbia complessita' = target.

    Perche' serve: se si sceglie D prima del clip (come faceva il vecchio codice),
    il clip cambia la complessita' e si ottengono piu' elementi del previsto.

    Come funziona:
        - senza clip vale C(D * M0) = D * C(M0): da qui una prima stima di D;
        - la complessita' cresce sempre con D, quindi si trova un intervallo
          [D_low, D_high] che contiene la soluzione e lo si dimezza (bisezione).

    Restituisce (M, D).
    """
    def complexity_for(D):
        M = clip_metric(D * M0, h_min, h_max, max_aspect_ratio)
        return metric_complexity(M, points, triangles)

    # --- il target si puo' raggiungere? -------------------------------------
    # Con h_max nessuna zona ha taglia > h_max: la complessita' minima e' area / h_max^2.
    area = triangle_areas(points, triangles).sum()
    if h_max is not None and target < (1 - tol) * area / h_max**2:
        raise ValueError(f"Complessita' {target} troppo bassa: con h_max = {h_max} "
                         f"il minimo possibile e' {area / h_max**2:.1f}")
    if h_min is not None and target > (1 + tol) * area / h_min**2:
        raise ValueError(f"Complessita' {target} troppo alta: con h_min = {h_min} "
                         f"il massimo possibile e' {area / h_min**2:.1f}")

    # --- 1. prima stima ------------------------------------------------------
    D = target / metric_complexity(M0, points, triangles)

    # --- 2. intervallo che contiene la soluzione ----------------------------
    D_low = D
    for _ in range(100):
        if complexity_for(D_low) <= target:
            break
        D_low = D_low / 2
    D_high = D
    for _ in range(100):
        if complexity_for(D_high) >= target:
            break
        D_high = D_high * 2

    # --- 3. bisezione --------------------------------------------------------
    for _ in range(200):
        # punto medio (in scala logaritmica)
        D = np.sqrt(D_low * D_high)
        C = complexity_for(D)
        if abs(C / target - 1) <= tol:
            break
        if C < target:
            D_low = D
        else:
            D_high = D

    M = clip_metric(D * M0, h_min, h_max, max_aspect_ratio)
    return M, float(D)


# =============================================================================
# Tutto insieme: metrica grezza dall'Hessiano di un campo
# =============================================================================

def hessian_metric(points, triangles, values, p=2.0, eigen_floor=1e-8):
    """Passi 1-3 in un colpo solo. Restituisce M0 e il numero di nodi senza Hessiano."""
    neighbors = node_neighbors(len(points), triangles)
    H, _ = hessian_lsq(points, values, neighbors)
    n_invalid = int(np.sum(~np.isfinite(H).all(axis=(1, 2))))
    M0 = metric_lp_unscaled(abs_hessian(H, eigen_floor), p)
    return M0, n_invalid


# =============================================================================
# 7. Adattamento goal-oriented (aggiunto del C_D)
# =============================================================================

def gradient_lsq(points, values, neighbors, max_condition=1e10):
    """Gradiente del campo `values` in ogni nodo, con lo stesso fit quadratico
    ai minimi quadrati (due anelli) di hessian_lsq.

    Restituisce:
        grad       (n, 2) = [d/dx, d/dy]; NaN dove il fit non e' affidabile
        condition  (n,) numero di condizionamento del sistema di ogni nodo
    """
    points = np.asarray(points, dtype=float)[:, :2]
    values = np.asarray(values, dtype=float)
    n = len(points)

    grad = np.full((n, 2), np.nan)
    condition = np.full(n, np.nan)

    for i in range(n):
        # vicini su due anelli, senza il nodo i
        stencil = set(neighbors[i])
        for j in neighbors[i]:
            stencil.update(neighbors[j])
        stencil.discard(i)
        if len(stencil) < 5:
            continue

        # distanze adimensionali
        ids = np.array(sorted(stencil))
        dx = points[ids, 0] - points[i, 0]
        dy = points[ids, 1] - points[i, 1]
        scale = np.max(np.sqrt(dx**2 + dy**2))
        dx = dx / scale
        dy = dy / scale

        # fit quadratico: beta = [gx, gy, Hxx, Hxy, Hyy] (adimensionali)
        A = np.column_stack([dx, dy, 0.5 * dx**2, dx * dy, 0.5 * dy**2])
        b = values[ids] - values[i]
        beta, _, rank, s = np.linalg.lstsq(A, b, rcond=None)

        if rank < 5 or s[-1] <= 0:
            continue
        condition[i] = s[0] / s[-1]
        if condition[i] > max_condition:
            continue

        # derivate prime: un solo fattore 1/scale (regola della catena)
        grad_x = beta[0] / scale
        grad_y = beta[1] / scale
        grad[i] = [grad_x, grad_y]

    return grad, condition


def euler_fluxes(rho, rho_u, rho_v, rho_E, gamma=1.4):
    """Flussi di Euler 2D dalle variabili conservative (array nodali).

    Restituisce Fx, Fy di forma (n, 4), colonna j = equazione j
    (0 massa, 1 q. di moto x, 2 q. di moto y, 3 energia), e la pressione p (n,).
    """
    u = rho_u / rho
    v = rho_v / rho
    p = (gamma - 1) * (rho_E - (rho_u**2 + rho_v**2) / 2 / rho)

    Fx = np.array([rho_u, rho_u * u + p, rho_v * u, u * (rho_E + p)]).T
    Fy = np.array([rho_v, rho_u * v, rho_v * v + p, v * (rho_E + p)]).T

    return Fx, Fy, p


def goal_oriented_hessian(points, triangles, Fx, Fy, grad_psi, eigen_floor=1e-8):
    """Hessiano goal-oriented (Loseille, Dervieux, Alauzet 2010):

        H_go = somma su j (equazioni) e k (direzioni) di |d psi_j / d x_k| * |H(F_kj)|

    Fx, Fy:    (n, 4) flussi di Euler (da euler_fluxes)
    grad_psi:  (n, 4, 2) gradienti dell'aggiunto:
               grad_psi[:, j, 0] = d psi_j / dx,  grad_psi[:, j, 1] = d psi_j / dy
    Restituisce H_go (n, 2, 2), simmetrica definita positiva.
    """
    n = len(points)
    # una volta sola per tutti gli 8 campi
    neighbors = node_neighbors(n, triangles)

    # gradiente non calcolato -> peso nullo
    grad_psi = np.nan_to_num(grad_psi)
    H_go = np.zeros((n, 2, 2))                   # accumulatore della somma

    for j in range(4):                           # equazioni: massa, q.d.m. x, q.d.m. y, energia
        # direzione x -> flusso Fx, direzione y -> flusso Fy
        for k, F in [(0, Fx), (1, Fy)]:
            # Hessiano del flusso F_kj
            H, _ = hessian_lsq(points, F[:, j], neighbors)
            Habs = abs_hessian(H, 0.0)                # |H|, NaN -> 0
            # |d psi_j / d x_k|, uno per nodo
            peso = np.abs(grad_psi[:, j, k])
            H_go = H_go + peso[:, None, None] * \
                Habs          # (n,1,1) * (n,2,2)

    # dove tutti i pesi sono ~0 (scia, psi = 0) H_go e' ~0: la soglia evita det = 0
    H_go = abs_hessian(H_go, eigen_floor)
    return H_go


def goal_oriented_metric(points, triangles, rho, rho_u, rho_v, rho_E, psi, gamma=1.4,
                         eigen_floor=1e-8):
    """Metrica goal-oriented non normalizzata M0, tutta insieme.

    psi: lista dei 4 campi aggiunti [psi_rho, psi_rho_u, psi_rho_v, psi_rho_E],
         nello stesso ordine delle equazioni.
    Usa p = 1: la stima dell'errore sul funzionale e' una norma L1.
    """
    Fx, Fy, _ = euler_fluxes(rho, rho_u, rho_v, rho_E, gamma)
    neighbors = node_neighbors(len(points), triangles)
    gradients = []
    for psi_j in psi:
        grad_j, _ = gradient_lsq(points, psi_j, neighbors)
        gradients.append(grad_j)                 # ciascuno (n, 2)
    grad_psi = np.stack(gradients, axis=1)       # (n, 4, 2)
    H_go = goal_oriented_hessian(
        points, triangles, Fx, Fy, grad_psi, eigen_floor)
    return metric_lp_unscaled(H_go, p=1)


def element_quality(points, triangles, M):
    """Qualita' UGAWG degli elementi nella metrica (Coppeans et al. 2026, eq. 12-14).

    points     (n, 2)     coordinate dei nodi
    triangles  (m, 3)     indici dei vertici
    M          (n, 2, 2)  metrica in ogni NODO
    Restituisce Q (m,): 1 = equilatero nella metrica, -> 0 = degenere.
    """
    V_hat = np.sqrt(3) / 4
    Q = np.zeros(len(triangles))

    for k, (a, b, c) in enumerate(triangles):
        # 1) tra i 3 vertici, la metrica con determinante massimo
        #    suggerimento: dets = np.linalg.det(M[[a, b, c]]) ; poi np.argmax

        # prende tutte le metriche su ogni noto del triangolo k
        Ms = M[[a, b, c]]
        dets = np.linalg.det(Ms)
        M_max = Ms[np.argmax(dets)]

        # 2) i tre lati come vettori: b-a, c-b, a-c

        v1 = points[b] - points[a]
        v2 = points[c] - points[b]
        v3 = points[a] - points[c]

        # 3) lunghezze al quadrato nella metrica: v @ M_max @ v  per ciascun lato

        l1 = v1.T @ M_max @ v1
        l2 = v2.T @ M_max @ v2
        l3 = v3.T @ M_max @ v3

        # 4) area euclidea del triangolo (puoi usare la formula del prodotto vettoriale)

        A = 1/2 * np.abs(v1[0]*v2[1] - v1[1]*v2[0])

        # 5) Q = (sqrt(det M_max) * area / V_hat) / (media delle tre lunghezze al quadrato)

        Q[k] = np.sqrt(np.max(dets)) * A / V_hat / \
            (1/3 * (l1**2 + l2**2 + l3**2))

    return Q
