pub const EYE_BOUND: [[i32; 8]; 2] = [
    [35, 41, 40, 42, 39, 37, 33, 36],
    [89, 95, 94, 96, 93, 91, 87, 90],
];

pub const HEAD_POSE_INDEX: [usize; 39] = [
    50, 51, 49, 48, 43, 102, 103, 104, 105, 101, 35, 41, 40, 42, 39, 37, 33, 36, 93, 96, 94, 95,
    89, 90, 87, 91, 75, 81, 84, 85, 80, 79, 78, 86, 61, 71, 52, 53, 0,
];

pub const HEAD_POSE_OBJECT: [[f32; 3]; 39] = [
    [1.330353, 7.122144, 6.903745],
    [2.533424, 7.878085, 7.451034],
    [4.861131, 7.878672, 6.601275],
    [6.137002, 7.271266, 5.200823],
    [6.825897, 6.760612, 4.402142],
    [-1.330353, 7.122144, 6.903745],
    [-2.533424, 7.878085, 7.451034],
    [-4.861131, 7.878672, 6.601275],
    [-6.137002, 7.271266, 5.200823],
    [-6.825897, 6.760612, 4.402142],
    [5.311432, 5.485328, 3.987654],
    [4.461908, 6.189018, 5.59441],
    [3.550622, 6.185143, 5.712299],
    [2.542231, 5.862829, 4.687939],
    [1.78993, 5.393625, 4.413414],
    [2.693583, 5.018237, 5.072837],
    [3.530191, 4.981603, 4.937805],
    [4.490323, 5.186498, 4.694397],
    [-5.311432, 5.485328, 3.987654],
    [-4.461908, 6.189018, 5.59441],
    [-3.550622, 6.185143, 5.712299],
    [-2.542231, 5.862829, 4.687939],
    [-1.78993, 5.393625, 4.413414],
    [-2.693583, 5.018237, 5.072837],
    [-3.530191, 4.981603, 4.937805],
    [-4.490323, 5.186498, 4.694397],
    [0.981972, 4.554081, 6.301271],
    [-0.981972, 4.554081, 6.301271],
    [-1.930245, 0.424351, 5.914376],
    [-0.746313, 0.348381, 6.263227],
    [0.0, 0.0, 6.76343],
    [0.746313, 0.348381, 6.263227],
    [1.930245, 0.424351, 5.914376],
    [0.0, 1.916389, 7.7],
    [-2.774015, -2.080775, 5.048531],
    [0.0, -1.646444, 6.704956],
    [2.774015, -2.080775, 5.048531],
    [0.0, -3.116408, 6.097667],
    [0.0, -7.415691, 4.070434],
];
