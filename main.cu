#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"

#define MAT_LAMBERTIAN 0
#define MAT_METAL 1
#define MAT_DIELECTRIC 2

struct Sphere // sphere struct to replace sphere : hitable class
{
    vec3 center;
    float radius;

    int mat_type; // see definitions above
    vec3 albedo;
    float fuzz;
    float ref_idx; // potentialy wasteful depending on material type, but we need all possible data
};

// helper function for the sphere hit quadratic calculation
__device__ bool hit_sphere(const Sphere &s, const ray &r, float t_min, float t_max, hit_record &rec)
{
    vec3 oc = r.origin() - s.center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - s.radius * s.radius;
    float discriminant = b * b - a * c;

    if (discriminant > 0)
    {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - s.center) / s.radius;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - s.center) / s.radius;
            return true;
        }
    }
    return false;
}

// helper function version of all of the old
__device__ bool scatter(const Sphere &s, const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state)
{
    if (s.mat_type == MAT_LAMBERTIAN) // lambertian scatter from material.h
    {
        vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
        scattered = ray(rec.p, target - rec.p);
        attenuation = s.albedo;
        return true;
    }
    else if (s.mat_type == MAT_METAL) // metal scatter logic
    {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + s.fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = s.albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }
    else if (s.mat_type == MAT_DIELECTRIC) // glass scatter logic
    {
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = vec3(1.0, 1.0, 1.0);
        vec3 refracted;
        float reflect_prob;
        float cosine;

        if (dot(r_in.direction(), rec.normal) > 0)
        {
            outward_normal = -rec.normal;
            ni_over_nt = s.ref_idx;
            cosine = s.ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        else
        {
            outward_normal = rec.normal;
            ni_over_nt = 1.0 / s.ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }

        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
        {
            reflect_prob = schlick(cosine, s.ref_idx);
        }
        else
        {
            reflect_prob = 1.0;
        }

        if (curand_uniform(local_rand_state) < reflect_prob)
        {
            scattered = ray(rec.p, reflected);
        }
        else
        {
            scattered = ray(rec.p, refracted);
        }
        return true;
    }
    return false;
}

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray &r, Sphere *spheres, int num_spheres, curandState *local_rand_state)
{
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);

    for (int i = 0; i < 50; i++)
    {
        hit_record rec;
        bool hit_anything = false;
        float closest_so_far = FLT_MAX;
        int hit_index = -1;

        // over ALL spheres to find the closest hit
        for (int i = 0; i < num_spheres; i++)
        {
            hit_record temp_rec;
            if (hit_sphere(spheres[i], cur_ray, 0.001f, closest_so_far, temp_rec))
            {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
                hit_index = i;
            }
        }

        if (hit_anything)
        {
            ray scattered;
            vec3 attenuation;
            // use the data from the hit sphere to calculate scatter
            if (scatter(spheres[hit_index], cur_ray, rec, attenuation, scattered, local_rand_state))
            {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else
            {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else
        {
            // the blue for hitting the sky
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0); // recurse
}

__global__ void rand_init(curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, Sphere *world, int num_spheres, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;

    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);

    for (int s = 0; s < ns; s++)
    {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, num_spheres, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(Sphere *d_spheres, int num_spheres, camera **d_camera, int nx, int ny, curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state = *rand_state;

        // Ground
        d_spheres[0].center = vec3(0, -1000.0, -1);
        d_spheres[0].radius = 1000;
        d_spheres[0].mat_type = MAT_LAMBERTIAN;
        d_spheres[0].albedo = vec3(0.5, 0.5, 0.5);

        int i = 1;
        for (int a = -11; a < 11; a++)
        {
            for (int b = -11; b < 11; b++)
            {
                float choose_mat = RND;
                vec3 center(a + RND, 0.2, b + RND);
                d_spheres[i].center = center;
                d_spheres[i].radius = 0.2;

                if (choose_mat < 0.8f)
                { // Lambertian
                    d_spheres[i].mat_type = MAT_LAMBERTIAN;
                    d_spheres[i].albedo = vec3(RND * RND, RND * RND, RND * RND);
                }
                else if (choose_mat < 0.95f)
                { // Metal
                    d_spheres[i].mat_type = MAT_METAL;
                    d_spheres[i].albedo = vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND));
                    d_spheres[i].fuzz = 0.5f * RND;
                }
                else
                { // Glass
                    d_spheres[i].mat_type = MAT_DIELECTRIC;
                    d_spheres[i].ref_idx = 1.5;
                }
                i++;
            }
        }

        // The 3 big spheres
        d_spheres[i].center = vec3(0, 1, 0);
        d_spheres[i].radius = 1.0;
        d_spheres[i].mat_type = MAT_DIELECTRIC;
        d_spheres[i].ref_idx = 1.5;
        i++;

        d_spheres[i].center = vec3(-4, 1, 0);
        d_spheres[i].radius = 1.0;
        d_spheres[i].mat_type = MAT_LAMBERTIAN;
        d_spheres[i].albedo = vec3(0.4, 0.2, 0.1);
        i++;

        d_spheres[i].center = vec3(4, 1, 0);
        d_spheres[i].radius = 1.0;
        d_spheres[i].mat_type = MAT_METAL;
        d_spheres[i].albedo = vec3(0.7, 0.6, 0.5);
        d_spheres[i].fuzz = 0.0;

        *rand_state = local_rand_state;

        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0;
        float aperture = 0.1;
        *d_camera = new camera(lookfrom,
                               lookat,
                               vec3(0, 1, 0),
                               30.0,
                               float(nx) / float(ny),
                               aperture,
                               dist_to_focus);
    }
}

__global__ void free_camera(camera **d_camera)
{
    delete *d_camera;
}

int main()
{
    int nx = 1200;
    int ny = 800;
    int ns = 10;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1, 1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables & the camera. test
    // hitable **d_list;
    Sphere *d_spheres;
    int num_spheres = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void **)&d_spheres, num_spheres * sizeof(Sphere)));
    // hitable **d_world;
    // checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1, 1>>>(d_spheres, num_spheres, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_spheres, num_spheres, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    std::cout << "P3\n"
              << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--)
    {
        for (int i = 0; i < nx; i++)
        {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99 * fb[pixel_index].r());
            int ig = int(255.99 * fb[pixel_index].g());
            int ib = int(255.99 * fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());

    free_camera<<<1, 1>>>(d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_spheres));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}
