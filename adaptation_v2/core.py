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
                     max_aspect_ratio=None, tol=1e-4, extra=None):
    """Cerca D tale che la metrica finale  M = clip(D * M0)  abbia complessita' = target.

    Perche' serve: se si sceglie D prima del clip (come faceva il vecchio codice),
    il clip cambia la complessita' e si ottengono piu' elementi del previsto.

    Come funziona:
        - senza clip vale C(D * M0) = D * C(M0): da qui una prima stima di D;
        - la complessita' cresce sempre con D, quindi si trova un intervallo
          [D_low, D_high] che contiene la soluzione e lo si dimezza (bisezione).

    extra: funzione opzionale M -> M applicata dopo il clip (es. il raffinamento
    imposto sugli spigoli). Se c'e', D viene scelto in modo che la complessita'
    FINALE, compresa questa modifica, sia uguale a target.

    Restituisce (M, D).
    """
    def final_metric(D):
        M = clip_metric(D * M0, h_min, h_max, max_aspect_ratio)
        return M if extra is None else extra(M)

    def complexity_for(D):
        return metric_complexity(final_metric(D), points, triangles)

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

    return final_metric(D), float(D)


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
            # |H| SENZA soglia, NaN -> 0
            Habs = abs_hessian(H, 0.0)
            #   (la soglia va applicata una sola volta, alla fine: se la si
            #    applicasse qui verrebbe moltiplicata per |grad psi| e
            #    raffinerebbe senza motivo le zone uniformi dove psi varia)
            # |d psi_j / d x_k|, uno per nodo
            peso = np.abs(grad_psi[:, j, k])
            H_go = H_go + peso[:, None, None] * \
                Habs          # (n,1,1) * (n,2,2)

    # dove tutti i pesi sono ~0 (scia, psi = 0) H_go e' ~0: la soglia evita det = 0
    H_go = abs_hessian(H_go, eigen_floor)
    return H_go


def spectral_radius_sum(points, triangles, rho, rho_u, rho_v, rho_E, gamma=1.4):
    """s_i = somma sulle facce della cella duale del nodo i di (|u.n| + c) |S|.

    E' la stessa quantita' con cui SU2 calcola il passo locale:
        dt_i = CFL * |V_i| / s_i     ->     |V_i| / dt_i = s_i / CFL
    Celle duali mediane: per ogni triangolo e ogni suo lato la faccia duale va dal
    punto medio del lato al baricentro; sui lati di bordo meta' lato per estremo.
    """
    pts = np.asarray(points)[:, :2]
    T = np.asarray(triangles)
    u, v = rho_u / rho, rho_v / rho
    p = (gamma - 1) * (rho_E - 0.5 * rho * (u**2 + v**2))
    c = np.sqrt(np.maximum(gamma * p / rho, 0.0))

    s = np.zeros(len(pts))
    # baricentri dei triangoli
    G = pts[T].mean(axis=1)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        i, j = T[:, a], T[:, b]
        # faccia duale (vettore)
        d = G - 0.5 * (pts[i] + pts[j])
        # normale * lunghezza
        nS = np.stack([d[:, 1], -d[:, 0]], axis=1)
        um, vm, cm = 0.5 * (u[i] + u[j]), 0.5 * \
            (v[i] + v[j]), 0.5 * (c[i] + c[j])
        lam = np.abs(um * nS[:, 0] + vm * nS[:, 1]) + \
            cm * np.hypot(nS[:, 0], nS[:, 1])
        np.add.at(s, i, lam)
        np.add.at(s, j, lam)

    edges, count = np.unique(np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]),
                                     axis=1), axis=0, return_counts=True)
    for i, j in edges[count == 1]:                            # lati di bordo
        t = pts[j] - pts[i]
        L = np.linalg.norm(t)
        n = np.array([t[1], -t[0]]) / L
        for k in (i, j):
            s[k] += 0.5 * L * (abs(u[k] * n[0] + v[k] * n[1]) + c[k])
    return s


def goal_oriented_metric(points, triangles, rho, rho_u, rho_v, rho_E, psi, gamma=1.4,
                         eigen_floor=1e-8, rescale_adjoint=True):
    """Metrica goal-oriented non normalizzata M0, tutta insieme.

    psi: lista dei 4 campi aggiunti [psi_rho, psi_rho_u, psi_rho_v, psi_rho_E],
         nello stesso ordine delle equazioni.
    Usa p = 1: la stima dell'errore sul funzionale e' una norma L1.

    rescale_adjoint: l'aggiunto scritto da SU2 (DISCRETE_ADJOINT) e' quello dell'iterazione
    di punto fisso U <- U - P^-1 R(U): vale  U_bar = P^T lambda, con P = |V|/dt + J.
    Il termine |V_i|/dt_i = s_i / CFL e' proporzionale alla taglia della cella, quindi
    U_bar ~ (taglia locale) * lambda. La stima dell'errore vuole lambda (l'aggiunto
    "puntuale"): lo si ottiene, a meno di una costante, dividendo per s_i.
    (Verificato sulla parete: U_bar * CFL / s rispetta la condizione dell'aggiunto
    continuo phi.n = n_x / q_inf.)
    """
    if rescale_adjoint:
        s = spectral_radius_sum(points, triangles, rho,
                                rho_u, rho_v, rho_E, gamma)
        psi = [psi_j / s for psi_j in psi]
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


# =============================================================================
# 8. Qualita' della mesh rispetto alla metrica (convenzioni UGAWG,
#    Coppeans, Fidkowski, Martins, JCP 2026, eq. 10-14)
# =============================================================================

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
        Ms = M[[a, b, c]]
        dets = np.linalg.det(Ms)
        M_max = Ms[np.argmax(dets)]

        # 2) i tre lati come vettori
        v1 = points[b] - points[a]
        v2 = points[c] - points[b]
        v3 = points[a] - points[c]

        # 3) lunghezze al quadrato nella metrica
        l1 = v1 @ M_max @ v1
        l2 = v2 @ M_max @ v2
        l3 = v3 @ M_max @ v3

        # 4) area euclidea del triangolo
        A = 0.5 * np.abs(v1[0] * v2[1] - v1[1] * v2[0])

        # 5) area nella metrica (normalizzata) / media dei lati al quadrato
        Q[k] = (np.sqrt(np.max(dets)) * A / V_hat) / ((l1 + l2 + l3) / 3)

    return Q


def mesh_edges(triangles):
    """Lati della mesh, ciascuno una sola volta: array (n_lati, 2) di indici di nodo."""
    e = np.vstack(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def metric_edge_lengths(points, edges, M):
    """Lunghezza di ogni lato nella metrica (UGAWG, eq. 10-11).

    L_a, L_b = lunghezza del lato con la metrica costante del nodo a e del nodo b;
    L = (L_a - L_b) / log(L_a / L_b)  (media logaritmica), oppure (L_a + L_b)/2
    se le due sono quasi uguali. In una mesh che rispetta la metrica L ~ 1.
    """
    a, b = edges[:, 0], edges[:, 1]
    v = points[b] - points[a]                              # (n_lati, 2)
    La = np.sqrt(np.einsum("ei,eij,ej->e", v, M[a], v))
    Lb = np.sqrt(np.einsum("ei,eij,ej->e", v, M[b], v))
    L = 0.5 * (La + Lb)
    diff = np.abs(La - Lb) > 1e-3
    L[diff] = (La[diff] - Lb[diff]) / np.log(La[diff] / Lb[diff])
    return L


def interpolate_nodal(old_points, old_triangles, values, new_points):
    """Interpola linearmente un campo nodale dalla mesh vecchia ai punti nuovi.

    values: (n_old, ...) per esempio la metrica (n_old, 2, 2).
    Per ogni punto nuovo si cerca il triangolo vecchio che lo contiene e si usano
    le coordinate baricentriche. I punti che non cadono in nessun triangolo
    (bordo curvo discretizzato in modo diverso) prendono il triangolo "piu'
    vicino" con le coordinate baricentriche riportate in [0, 1].
    """
    old_points = np.asarray(old_points, dtype=float)
    new_points = np.asarray(new_points, dtype=float)
    P = old_points[old_triangles]                          # (m, 3, 2)

    # griglia di "secchi": ogni triangolo viene registrato nei secchi che tocca
    lo, hi = old_points.min(axis=0), old_points.max(axis=0)
    n_cells = max(1, int(np.sqrt(len(old_triangles) / 4)))
    size = (hi - lo) / n_cells
    tri_lo = np.floor((P.min(axis=1) - lo) /
                      size).astype(int).clip(0, n_cells - 1)
    tri_hi = np.floor((P.max(axis=1) - lo) /
                      size).astype(int).clip(0, n_cells - 1)
    buckets = {}
    for t in range(len(old_triangles)):
        for i in range(tri_lo[t, 0], tri_hi[t, 0] + 1):
            for j in range(tri_lo[t, 1], tri_hi[t, 1] + 1):
                buckets.setdefault((i, j), []).append(t)

    # coordinate baricentriche: x = A + l1 (B - A) + l2 (C - A)
    A, B, C = P[:, 0], P[:, 1], P[:, 2]
    T = np.stack([B - A, C - A], axis=2)                   # (m, 2, 2)
    T_inv = np.linalg.inv(T)

    result = np.empty((len(new_points),) + values.shape[1:])
    cell = np.floor((new_points - lo) / size).astype(int).clip(0, n_cells - 1)
    for k, x in enumerate(new_points):
        cand = []
        ring = 0
        while not cand and ring <= n_cells:            # allarga la ricerca se il secchio e' vuoto
            for i in range(cell[k, 0] - ring, cell[k, 0] + ring + 1):
                for j in range(cell[k, 1] - ring, cell[k, 1] + ring + 1):
                    cand.extend(buckets.get((i, j), []))
            ring += 1
        cand = np.array(cand)
        l12 = np.einsum("tij,tj->ti", T_inv[cand], x - A[cand])
        lam = np.column_stack([1 - l12.sum(axis=1), l12])  # (n_cand, 3)
        # dentro: min >= 0; fuori: il "meno fuori"
        best = np.argmax(lam.min(axis=1))
        w = np.clip(lam[best], 0, None)
        w = w / w.sum()
        result[k] = np.tensordot(w, values[old_triangles[cand[best]]], axes=1)
    return result


def mesh_quality_report(old_points, old_triangles, M, new_points, new_triangles):
    """Conformita' della mesh nuova alla metrica richiesta (calcolata sulla mesh vecchia).

    Restituisce un dizionario con le statistiche per il log.
    """
    M_new = interpolate_nodal(old_points, old_triangles, M, new_points)
    Q = element_quality(new_points, new_triangles, M_new)
    L = metric_edge_lengths(new_points, mesh_edges(new_triangles), M_new)
    in_range = (L >= np.sqrt(2) / 2) & (L <= np.sqrt(2))
    return {
        "mesh_complexity": round(float(np.sqrt(3) / 4 * len(new_triangles)), 1),
        "q_min": round(float(Q.min()), 3),
        "q_median": round(float(np.median(Q)), 3),
        "q_below_04_pct": round(100 * float(np.mean(Q < 0.4)), 2),
        "edges_unit_pct": round(100 * float(np.mean(in_range)), 2),
    }


# =============================================================================
# 9. Raffinamento imposto vicino ai punti singolari (spigoli del profilo)
# =============================================================================

# spigoli del diamante t/c = 0.07: bordo d'attacco, bordo d'uscita, spigoli a meta' corda
DIAMOND_CORNERS = [(0.0, 0.0), (1.0, 0.0), (0.5, 0.035), (0.5, -0.035)]


def corner_refinement(points, M, corners, h_corner, grading=0.3):
    """Impone una taglia isotropa piccola vicino agli spigoli.

    Taglia richiesta a distanza d dallo spigolo piu' vicino:
        h(d) = h_corner + grading * d
    (h_corner sullo spigolo, poi cresce linearmente: grading = 0.3 vuol dire
    che la taglia aumenta di 0.3 per ogni unita' di distanza, cioe' circa del
    30 % da un elemento al successivo).

    La nuova metrica e' l'intersezione tra M e la metrica isotropa I / h(d)^2:
    nella base degli autovettori di M ogni autovalore diventa
        max(lambda, 1 / h(d)^2)
    cioe' si prende, direzione per direzione, la taglia piu' piccola delle due.
    Lontano dagli spigoli h(d) e' grande e M non cambia.

    Restituisce (M nuova, numero di nodi in cui M e' cambiata).
    """
    points = np.asarray(points)[:, :2]
    corners = np.asarray(corners, dtype=float)
    d = np.min(np.linalg.norm(
        points[:, None, :] - corners[None, :, :], axis=2), axis=1)
    h = h_corner + grading * d
    lam_corner = 1.0 / h ** 2

    eigenvalues, eigenvectors = np.linalg.eigh(M)
    changed = (eigenvalues < lam_corner[:, None]).any(axis=1)
    new_eigenvalues = np.maximum(eigenvalues, lam_corner[:, None])
    return from_eig(new_eigenvalues, eigenvectors), int(changed.sum())
