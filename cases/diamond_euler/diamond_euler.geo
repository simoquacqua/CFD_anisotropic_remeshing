// DIAMOND AIRFOIL — baseline Euler 2D, geometria senza adattamento agli urti
// c=1, t/c=0.07. Il flusso va da sinistra (LE) a destra (TE).
// Eseguire: gmsh diamond_euler.geo -2 -format su2 -o diamond_euler.su2

// Parametri geometrici (lunghezze in unità di corda)
c = 1.0;
tc = 0.07;
t = c * tc;
Rfar = 5.0 * c;           // raggio iniziale: verificare in seguito sensibilità dominio

// Parametri mesh (nessuna conoscenza delle posizioni degli urti)
hWall = 0.02 * c;         // taglia bersaglio vicino al profilo
hFar  = 0.30 * c;         // taglia bersaglio lontano dal profilo

// Diamond: LE, vertice superiore, TE, vertice inferiore
Point(1) = {0.0,   0.0,  0.0, hWall};       // LE
Point(2) = {c/2,  t/2,  0.0, hWall};       // vertice superiore
Point(3) = {c,     0.0, 0.0, hWall};       // TE
Point(4) = {c/2, -t/2,  0.0, hWall};       // vertice inferiore

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};

// Bordo esterno circolare, centrato a metà corda
Point(5) = {c/2,         0.0,   0.0, hFar}; // centro
Point(6) = {c/2 + Rfar,  0.0,   0.0, hFar}; // destra
Point(7) = {c/2,         Rfar,  0.0, hFar}; // alto
Point(8) = {c/2 - Rfar,  0.0,   0.0, hFar}; // sinistra
Point(9) = {c/2,        -Rfar,  0.0, hFar}; // basso

Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};
Curve Loop(2) = {5, 6, 7, 8};

// Dominio fluido: cerchio esterno meno il foro del profilo
Plane Surface(1) = {2, 1};

// Marker che devono coincidere ESATTAMENTE con il file SU2 .cfg
Physical Surface("Domain") = {1};
Physical Curve("Airfoil")  = {1, 2, 3, 4};
Physical Curve("Farfield") = {5, 6, 7, 8};

// Sola gradazione geometrica verso la parete, NON verso gli urti
Field[1] = Distance;
Field[1].CurvesList = {1, 2, 3, 4};
Field[1].Sampling = 200;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = hWall;
Field[2].SizeMax = hFar;
Field[2].DistMin = 0.03 * c;
Field[2].DistMax = 1.00 * c;
Background Field = 2;

Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeMin = hWall;
Mesh.MeshSizeMax = hFar;
Mesh.Algorithm = 5;      // Delaunay (triangoli)

// Niente 'Mesh 2' o 'Save' qui: la generazione e l'export avvengono da terminale.
