#include <iostream>
#include <glm/glm.hpp>
#include <SDL.h>
#include "SDLauxiliary.h"
#include "TestModel.h"
#include <omp.h>
#include "LoadSTL.cpp"

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec2;
using glm::ivec2;
using glm::vec4;
using glm::mat4;

/* ----------------------------------------------------------------------------*/
/* GLOBAL VARIABLES                                                            */

//#define CUSTOM_MODEL

const int SCREEN_WIDTH = 500;
const int SCREEN_HEIGHT = 500;
SDL_Surface* screen;
int t;
float focalLength = 500.0f;

vec3 cameraPos( 0, 0, -3.0f );
mat3 cameraRot = mat3(0.0f);
float yaw = 0; // Yaw angle controlling camera rotation around y-axis

vec3 currentColor;
vec3 currentNormal;
vec3 currentReflectance;

float depthBuffer[SCREEN_HEIGHT][SCREEN_WIDTH];

int NUM_LIGHTS = 0;
Light lights[32];

vec3 indirectLightPowerPerArea = 0.2f*vec3( 1, 1, 1 );
bool isUpdated = true;

bool MULTITHREADING_ENABLED = false;
int NUM_THREADS; // Set by code
int SAVED_THREADS; // Stores thread value when changed

bool BACKFACE_CULLING_ENABLED = false;
bool FRUSTUM_CULLING_ENABLED = true;

bool DOF_ENABLED = false;
int DOF_KERNEL_SIZE = 8;
float FOCAL_LENGTH = 1.3f;

/* KEY STATES                                                                  */
bool OMP_key_pressed = false;
bool thread_add_key_pressed = false;
bool thread_subtract_key_pressed = false;
bool backface_key_pressed = false;
bool frustum_key_pressed = false;
bool delete_light_key_pressed = false;
bool add_light_key_pressed = false;
bool DOF_key_pressed = false;

/* ----------------------------------------------------------------------------*/
/* FUNCTIONS                                                                   */
vector<Triangle> triangles;
vector<Triangle> activeTriangles;

// Depth of field data containers
float focalDistances[SCREEN_WIDTH * SCREEN_HEIGHT];
vec3 pixelColours[SCREEN_WIDTH * SCREEN_HEIGHT];
vec3 blurredPixels[SCREEN_WIDTH * SCREEN_HEIGHT];

// Clipping volume bounds
float minX = -1.0f;
float minY = -1.0f;
float minZ = 0.0f;
float maxX = 1.0f;
float maxY = 1.0f;
float maxZ = 1.0f;

void Update();
void Draw();
void VertexShader( const vec3& v, Pixel& p );
void Interpolate( Pixel a, Pixel b, vector<Pixel>& result );
void Breshenham(Pixel a, Pixel b, vector<Pixel>& result);
void DrawLineSDL( SDL_Surface* surface, Pixel a, Pixel b, vec3 color );
void ComputePolygonRows( const vector<Pixel>& vertexPixels, vector<Pixel>& leftPixels, vector<Pixel>& rightPixels , vec3 color, vec3 normal);
void DrawRows( const vector<Pixel>& leftPixels, const vector<Pixel>& rightPixels , vec3 color, vec3 normal);
void DrawPolygon( const vector<Vertex>& vertices , vec3 color, vec3 normal);
void PixelShader( const Pixel& p , vec3 color, vec3 normal);
void AddLight(vec3 position, vec3 color, float intensity);
void DeleteLight();
float RandomNumber();
void CalculateDOF();
bool InCuboid(vec4 v);

int main( int argc, char* argv[] )
{
	screen = InitializeSDL( SCREEN_WIDTH, SCREEN_HEIGHT );
	AddLight(vec3(0, -0.5f, -0.7f), vec3(1,1,1), 14 );

	#ifdef CUSTOM_MODEL
		LoadSTL customModel;
		customModel.LoadSTLFile(triangles);
		cameraPos = vec3(0,-0.5,-5.0f);
	#else
		// Generate the Cornell Box
		LoadTestModel( triangles );
	#endif

	cameraRot[1][1] = 1.0f;

	// Request as many threads as the system can provide
	NUM_THREADS = omp_get_max_threads();
    omp_set_num_threads(NUM_THREADS);


    // Set NUM_THREADS to how many the system can actually provide
    #pragma omp parallel
    {
    	int ID = omp_get_thread_num();
    	if(ID == 0)
    		NUM_THREADS = omp_get_num_threads();
    	    SAVED_THREADS = NUM_THREADS;
    }

    if(MULTITHREADING_ENABLED)
    {
    	cout << "Multithreading enabled with " << NUM_THREADS << " threads" << endl;
    }
    else
    	omp_set_num_threads(1);

	t = SDL_GetTicks();	// Set start value for timer.

	while( NoQuitMessageSDL() )
	{
			Update();
			if (isUpdated)
			{
				Draw();
				isUpdated = false;
			}
	}

	SDL_SaveBMP( screen, "screenshot.bmp" );
	return 0;
}

// Returns a random number between -0.5 and 0.5
float RandomNumber()
{
	return ((double) rand() / (RAND_MAX)) - 0.5f;
}

void AddLight(vec3 position, vec3 color, float intensity)
{
	lights[NUM_LIGHTS].position = position;
	lights[NUM_LIGHTS].color = color;
	lights[NUM_LIGHTS].intensity = intensity;

	NUM_LIGHTS++;
}

void DeleteLight()
{
	if(NUM_LIGHTS > 0)
		NUM_LIGHTS--;
}

void Update()
{
	// Compute frame time:
	int t2 = SDL_GetTicks();
	float dt = float(t2-t);
	t = t2;
	cout << "Render time: " << dt << " ms." << endl;

	// Clear the screen before drawing the next frame
	#pragma omp parallel for schedule(auto)
	for( int y=0; y<SCREEN_HEIGHT; ++y )
	{
		for( int x=0; x<SCREEN_WIDTH; ++x )
		{
			vec3 color( 0.0f, 0.0f, 0.0f );
			depthBuffer[y][x] = 0.0f;
			PutPixelSDL( screen, x, y, color );
		}
	}

	// Adjust camera transform
	vec3 right(cameraRot[0][0], cameraRot[0][1], cameraRot[0][2]);
	vec3 down(cameraRot[1][0], cameraRot[1][1], cameraRot[1][2]);
	vec3 forward(cameraRot[2][0], cameraRot[2][1], cameraRot[2][2]);
	Uint8* keystate = SDL_GetKeyState( 0 );

	if(!OMP_key_pressed && keystate[SDLK_4])
	{
		MULTITHREADING_ENABLED = !MULTITHREADING_ENABLED;
		if(!MULTITHREADING_ENABLED)
			NUM_THREADS = 1;
		else
			NUM_THREADS = SAVED_THREADS;
		omp_set_num_threads(NUM_THREADS);
		cout << "Multithreading toggled to " << MULTITHREADING_ENABLED << endl;
		OMP_key_pressed = true;
	}
	else if (!keystate[SDLK_4])
		OMP_key_pressed = false;

	if(!thread_subtract_key_pressed && keystate[SDLK_5])
	{
		NUM_THREADS--;
		SAVED_THREADS = NUM_THREADS;
		omp_set_num_threads(NUM_THREADS);
		cout << "Threads decreased to " << NUM_THREADS << endl;
		thread_subtract_key_pressed = true;
		isUpdated = true;
	}
	else if (!keystate[SDLK_5])
		thread_subtract_key_pressed = false;

	if(!thread_add_key_pressed && keystate[SDLK_6])
	{
		NUM_THREADS++;
		SAVED_THREADS = NUM_THREADS;
		omp_set_num_threads(NUM_THREADS);
		cout << "Threads increased to " << NUM_THREADS << endl;
		thread_add_key_pressed = true;
		isUpdated = true;
	}
	else if (!keystate[SDLK_6])
		thread_add_key_pressed = false;

	if(!backface_key_pressed && keystate[SDLK_7])
	{
		if(BACKFACE_CULLING_ENABLED)
		{
			BACKFACE_CULLING_ENABLED = false;
			// Reactive all triangles
			for( size_t i = 0; i < triangles.size(); ++i )
			{
				triangles[i].isCulled = false;
			}
		}
		else
			BACKFACE_CULLING_ENABLED = true;
		cout << "Backface culling toggled " << endl;
		backface_key_pressed = true;
		isUpdated = true;
	}
	else if (!keystate[SDLK_7])
		backface_key_pressed = false;

	if(!frustum_key_pressed && keystate[SDLK_8])
	{
		if(FRUSTUM_CULLING_ENABLED)
		{
			FRUSTUM_CULLING_ENABLED = false;
			// Reactive all triangles
			for( size_t i = 0; i < triangles.size(); ++i )
			{
				triangles[i].isCulled = false;
			}
		}
		else
			FRUSTUM_CULLING_ENABLED = true;
		cout << "Frustum culling toggled " << endl;
		frustum_key_pressed = true;
		isUpdated = true;
	}
	else if (!keystate[SDLK_8])
		frustum_key_pressed = false;

	if(!add_light_key_pressed && keystate[SDLK_2])
	{
		AddLight(vec3(RandomNumber() * 2.0f, RandomNumber() * 2.0f, RandomNumber() * 2.0f),vec3(abs(RandomNumber()) * 2.0f + 0.2f,abs(RandomNumber()) * 2.0f + 0.2f,abs(RandomNumber()) * 2.0f + 0.2f),abs(RandomNumber()) * 20.0f);
		cout << "Spawned a light" << endl;
		add_light_key_pressed = true;
		isUpdated = true;
	}
	else if (!keystate[SDLK_2])
	{
		add_light_key_pressed = false;
	}

	if(!delete_light_key_pressed && keystate[SDLK_3])
	{
		DeleteLight();
		cout << "Deleted a light" << endl;
		delete_light_key_pressed = true;
		isUpdated = true;
	}
	else if (!keystate[SDLK_3])
	{
		delete_light_key_pressed = false;
	}

	if(!DOF_key_pressed && keystate[SDLK_9])
	{
		DOF_ENABLED = !DOF_ENABLED;
		cout << "Depth of Field toggled to " << DOF_ENABLED << endl;
		DOF_key_pressed = true;
		isUpdated = true;
	}
	else if (!keystate[SDLK_9])
	{
		DOF_key_pressed = false;
	}

	if (keystate[SDLK_RIGHTBRACKET])
	{
		FOCAL_LENGTH += 0.1f;
		cout << "Focal length is " << FOCAL_LENGTH << endl;
		isUpdated = true;
	}
	else if (keystate[SDLK_LEFTBRACKET])
	{
		FOCAL_LENGTH -= 0.1f;
		cout << "Focal length is " << FOCAL_LENGTH << endl;
		isUpdated = true;
	}

	if( keystate[SDLK_UP] )
	{
		// Move camera forward
		cameraPos += 0.05f*forward*(dt / 20.0f);
		isUpdated = true;
	}
	else if( keystate[SDLK_DOWN] )
	{
		// Move camera backward
		cameraPos -= 0.05f*forward*(dt / 20.0f);
		isUpdated = true;
	}
	if( keystate[SDLK_LEFT] )
	{
		// Rotate camera to the left
		yaw += 0.01f*(dt / 20.0f);
		isUpdated = 1;
	}
	else if( keystate[SDLK_RIGHT] )
	{
		// Rotate camera to the right
		yaw -= 0.01f*(dt / 20.0f);
		isUpdated = true;
	}

	// Light movement controls
	if (keystate[SDLK_w])
	{
		lights[0].position.z += 0.05f*(dt / 20.0f);
		isUpdated = true;
	}
	else if (keystate[SDLK_s])
	{
		lights[0].position.z -= 0.05f*(dt / 20.0f);
		isUpdated = true;
	}

	if (keystate[SDLK_a])
	{
		lights[0].position.x -= 0.05f*(dt / 20.0f);
		isUpdated = true;;
	}
	else if (keystate[SDLK_d])
	{
		lights[0].position.x += 0.05f*(dt / 20.0f);
		isUpdated = true;
	}

	if (isUpdated)
	{
		// Update camera rotation matrix
		float c = cos(yaw);
		float s = sin(yaw);
		cameraRot[0][0] = c;
		cameraRot[0][2] = s;
		cameraRot[2][0] = -s;
		cameraRot[2][2] = c;

		vec3 fVec = glm::normalize(vec3(0,0,1.0f)*cameraRot);
		float near = cameraPos.z+fVec.z*2.5f, far = cameraPos.z+fVec.z*15.0f;
		//float near = 3.0f, far = 10.0f;
		float w = (float)SCREEN_WIDTH, h = (float)SCREEN_HEIGHT;

		// Perspective matrix transformation
		mat4 transform = glm::mat4(0.0f);
		// fovy version
		vec3 t(0.0f, -h/2.0f, focalLength);
		vec3 b(0.0f, h/2.0f, focalLength);
		float cy = dot(t,b)/(glm::length(t)*glm::length(b));
		float rfovy = acos(cy);
		float fovy = (180.0f/M_PI)*rfovy;
		float aspect = w/h;
		transform[0][0] = (1.0f/tan(rfovy/2.0f))/aspect;
		transform[1][1] = (1.0f/tan(rfovy/2.0f));
		transform[2][2] = far/(far-near);
		transform[3][2] = near*far/(far-near);
		transform[3][2] = 1.0f;

		for( size_t i = 0; i < triangles.size(); ++i )
		{
			// Backface culling
			if (BACKFACE_CULLING_ENABLED)
			{
				if (dot(forward,triangles[i].normal) > 0.8f)
				{
					triangles[i].isCulled = true;
				}
				else
					triangles[i].isCulled = false;
			}

			if (FRUSTUM_CULLING_ENABLED)
			{
				// Get every vertex of triangle to camera space
				vec3 v0 = triangles[i].v0;
				vec3 v1 = triangles[i].v1;
				vec3 v2 = triangles[i].v2;

				v0 = (v0-cameraPos)*cameraRot;
				v1 = (v1-cameraPos)*cameraRot;
				v2 = (v2-cameraPos)*cameraRot;

				// Map to clipping space
				vec4 tv0 = glm::vec4(v0.x, v0.y, v0.z, 1.0f);
				vec4 tv1 = glm::vec4(v1.x, v1.y, v1.z, 1.0f);
				vec4 tv2 = glm::vec4(v2.x, v2.y, v2.z, 1.0f);
				tv0 = tv0*transform;
				tv1 = tv1*transform;
				tv2 = tv2*transform;

				tv0 = tv0/tv0[3];
				tv1 = tv1/tv1[3];
				tv2 = tv2/tv2[3];

				bool bv0 = InCuboid(tv0);
				bool bv1 = InCuboid(tv1);
				bool bv2 = InCuboid(tv2);

				// Determine culling
				if (bv0 || bv1 || bv2)
					triangles[i].isCulled = false;
				else
					triangles[i].isCulled = true;
			}
		}
	}
}

bool InCuboid(glm::vec4 v)
{
	if (v.x >= minX && v.x <= maxX && v.y >= minY && v.y <= maxY && v.z >= minZ && v.z <= maxZ)
	{
		return true;
	}
	return false;
}


void Draw()
{
	if( SDL_MUSTLOCK(screen) )
		SDL_LockSurface(screen);

	currentReflectance = vec3(1.0f,1.0f,1.0f);

	for( size_t i = 0; i < triangles.size(); ++i )
	{
		if (!triangles[i].isCulled)
		{
			// Get the 3 vertices of the triangle
			vector<Vertex> vertices(3);
			vertices[0].position = triangles[i].v0;
			vertices[1].position = triangles[i].v1;
			vertices[2].position = triangles[i].v2;
			DrawPolygon( vertices , triangles[i].color, triangles[i].normal);
		}
	}

		if( SDL_MUSTLOCK(screen) )
		SDL_UnlockSurface(screen);
		

	SDL_UpdateRect( screen, 0, 0, 0, 0 );

	//CalculateDOF();
}

void CalculateDOF()
{
	if( SDL_MUSTLOCK(screen) )
		SDL_LockSurface(screen);

	// Total number of pixels in the kernel
	float totalPixels = DOF_KERNEL_SIZE * DOF_KERNEL_SIZE;

	// Spawn threads
	#pragma omp parallel for schedule(auto)
	for (int y = 1; y < SCREEN_HEIGHT - 1; y++)
	{
		for (int x = 1; x < SCREEN_WIDTH - 1; x++)
		{
			vec3 finalColour(0.0f,0.0f,0.0f);
			if(DOF_ENABLED)
			{
				// Start from top left of kernel
				for(int z = ceil(DOF_KERNEL_SIZE / -2.0f); z < ceil(DOF_KERNEL_SIZE / 2.0f); z++)
				{
					for(int z2 = ceil(DOF_KERNEL_SIZE / -2.0f); z2 < ceil(DOF_KERNEL_SIZE / 2.0f); z2++)
					{
						float weighting;
						if(z == 0 && z2 == 0)
							weighting = 1 - (min(abs(focalDistances[y*SCREEN_HEIGHT+x]), 1.0f) * ((totalPixels - 1) / totalPixels) );
						else
							weighting = min(abs(focalDistances[y*SCREEN_HEIGHT+x]), 1.0f) * (1.0f / totalPixels);

						// Add contribution to final pixel colour
						finalColour += pixelColours[(y+z)*SCREEN_HEIGHT+(x+z2)] * weighting;
					}
				}
			}
			else
			{
				finalColour = pixelColours[y*SCREEN_HEIGHT+x];
			}

			PutPixelSDL( screen, x, y, finalColour );
		}
	}

	if( SDL_MUSTLOCK(screen) )
		SDL_UnlockSurface(screen);
		

	SDL_UpdateRect( screen, 0, 0, 0, 0 );

}

// Compute 2D screen position from 3D position
void VertexShader( const Vertex& v, Pixel& p )
{
	// We need to adjust the vertex positions based on the camera position and rotation
	vec3 pos = (v.position - cameraPos) * cameraRot;

	// Store the 3D position of the vertex in the pixel. Divide by the Z value so the value interpolates correctly over the perspective
	p.pos3d = pos / pos.z;

	// Calculate depth of pixel, inversed so the value interpolates correctly over the perspective
	p.zinv = 1.0f / pos.z;

	// Calculate 2D screen position and place (0,0) at top left of the screen
	p.x = int(focalLength * (pos.x * p.zinv)) + (SCREEN_WIDTH / 2.0f);
	p.y = int(focalLength * (pos.y * p.zinv)) + (SCREEN_HEIGHT / 2.0f);
}

// Calculate per pixel lighting
void PixelShader( const Pixel& p , vec3 color, vec3 normal)
{
	int x = p.x;
	int y = p.y;

	// Multiply pixel 3d position by the z value to get the original position from the inverse
	vec3 pPos3d(p.pos3d);
	//pPos3d *= p.pos3d.z;
	pPos3d /=p.zinv;
	// Invert the camera transformations of pos3d P'=(P-C)R
	pPos3d = pPos3d*glm::inverse(cameraRot);
	pPos3d+=cameraPos;

	vec3 result(0.0f, 0.0f, 0.0f);

	//float distance = glm::distance(pPos3d, cameraPos);
	//focalDistances[p.y*SCREEN_HEIGHT + p.x] = distance - FOCAL_LENGTH;

	for(int i = 0; i < NUM_LIGHTS; i++)
	{
		// Apply pos3d transform to light
		vec3 lightPos = lights[i].position;
		/*lightPos = (lightPos-cameraPos)*cameraRot;
		lightPos = lightPos / lightPos.z;*/

		// Calculate lighting
		float r = glm::distance(pPos3d, lightPos);
		float A = 4*M_PI*(r*r);
		vec3 lightColor = lights[i].color * lights[i].intensity;
		vec3 rDir = glm::normalize(lightPos - pPos3d);
		vec3 nDir = normal;
		vec3 B = lightColor / A;

		vec3 D = (B * max(glm::dot(rDir,nDir), 0.0f));
		result += D;
	}


	vec3 pixelColor = currentReflectance * (result + indirectLightPowerPerArea) * color;
	//pixelColours[y*SCREEN_HEIGHT + x] = pixelColor;

	PutPixelSDL( screen, x, y, pixelColor );
}

// Draws a line between two points
void DrawLineSDL( SDL_Surface* surface, Pixel a, Pixel b, vec3 color, vec3 normal)
{
	Pixel delta = a - b;
	PixelAbs(delta);
	
	// DDA
	/*int pixels = max( delta.x, delta.y ) + 1;

	vector<Pixel> line( pixels );

	Interpolate( a, b, line );*/

	// Breshenam
	int pixels = b.x-a.x;
	vector<Pixel> line (pixels);

	Breshenham(a,b,line);

	// Spawn threads
	#pragma omp parallel for schedule(auto)
	for(int i = 0; i < pixels; ++i)
	{
		// Ensure pixel is on the screen and is closer to the camera than the current value in the depth buffer
		if(line[i].y < SCREEN_HEIGHT && line[i].y >= 0 && line[i].x < SCREEN_WIDTH && line[i].x >= 0 && line[i].zinv > depthBuffer[line[i].y][line[i].x])
		{
			depthBuffer[line[i].y][line[i].x] = line[i].zinv;
			PixelShader(line[i], color, normal);
		}
	}
}

// Interpolates between two Pixels
void Interpolate( Pixel a, Pixel b, vector<Pixel>& result )
{
	int N = result.size();
	Pixel delta = b-a;

	fPixel step(delta);

	step = (step / float(max(N-1,1)));

	fPixel current( a );

	for( int i=0; i<N; ++i )
	{
		result[i].x = current.x;
		result[i].y = current.y;
		result[i].zinv = current.zinv;
		result[i].pos3d = current.pos3d;
		current.x += step.x;
		current.y += step.y;
		current.zinv += step.zinv;
		current.pos3d += step.pos3d;
	}
}

void Breshenham(Pixel a, Pixel b, vector<Pixel>& result)
{
	int x = a.x;
	int y = a.y;
	int dx = b.x-a.x, dy = b.y-a.y;
	int dx2 = 2*dx, dy2 = 2*dy;
	int dydx2 = dy2 - dx2;
	int d = dy2 - dx;

	float zinv = (b.zinv - a.zinv)/float(dx);
	vec3 pos3d = (b.pos3d - a.pos3d)/float(dx);

	for (int i = 0;  i < dx; i++)
	{
		x+=1;
		if (d<0)
		{
			d+=dy2;
		}
		else
		{
			y+=1;
			d+=dydx2;
		}
		result[i].x = x;
		result[i].y = y;
		result[i].zinv = a.zinv+zinv*float(i);
		result[i].pos3d = a.pos3d+pos3d*float(i);
	}
}

void ComputePolygonRows( const vector<Pixel>& vertexPixels, vector<Pixel>& leftPixels, vector<Pixel>& rightPixels )
{
	// 1. Find max and min y-value of the polygon
	// and compute the number of rows it occupies.

	int maxY = max(max(vertexPixels[0].y, vertexPixels[1].y), vertexPixels[2].y);
	int minY = min(min(vertexPixels[0].y, vertexPixels[1].y), vertexPixels[2].y);

	int ROWS = maxY - minY + 1;

	// 2. Resize leftPixels and rightPixels
	// so that they have an element for each row.

	leftPixels.resize( ROWS );
	rightPixels.resize( ROWS );

	// 3. Initialize the x-coordinates in leftPixels
	// to some really large value and the x-coordinates
	// in rightPixels to some really small value.

	for( int i = 0; i < ROWS; ++i )
	{
		leftPixels[i].x = +numeric_limits<int>::max();
		rightPixels[i].x = -numeric_limits<int>::max();
	}

	// 4. Loop through all edges of the polygon and use
	// linear interpolation to find the x-coordinate for
	// each row it occupies. Update the corresponding
	// values in rightPixels and leftPixels.

	for(int i = 0; i < 3; i++)
	{
		int j = (i + 1) % 3; // Ensure all 3 edges are looped through
		// Adjust vertex positions to have y value 0 at minY so Y coordinates map to array indicies
		Pixel v1 (vertexPixels[i].x, vertexPixels[i].y - minY, vertexPixels[i].zinv, vertexPixels[i].pos3d);
		Pixel v2 (vertexPixels[j].x, vertexPixels[j].y - minY, vertexPixels[j].zinv, vertexPixels[j].pos3d);

		int edgePixels = abs(vertexPixels[i].y - vertexPixels[j].y) + 1; // Calculate number of rows this edge occupies
		vector<Pixel> edgeResult(edgePixels); // Create array of ivec2 with number of rows
		Interpolate(v1,v2,edgeResult); // Interpolate between the two vertices

		for(int k = 0; k < edgePixels; k++)
		{
			if(edgeResult[k].x < leftPixels[edgeResult[k].y].x)
			{
				leftPixels[edgeResult[k].y].x = edgeResult[k].x;
				leftPixels[edgeResult[k].y].y = edgeResult[k].y + minY;
				leftPixels[edgeResult[k].y].zinv = edgeResult[k].zinv;
				leftPixels[edgeResult[k].y].pos3d = edgeResult[k].pos3d;
			}

			if(edgeResult[k].x > rightPixels[edgeResult[k].y].x)
			{
				rightPixels[edgeResult[k].y].x = edgeResult[k].x;
				rightPixels[edgeResult[k].y].y = edgeResult[k].y + minY;
				rightPixels[edgeResult[k].y].zinv = edgeResult[k].zinv;
				rightPixels[edgeResult[k].y].pos3d = edgeResult[k].pos3d;
			}
		}
	}
}

// Draw a line for each row of the triangle
void DrawRows( const vector<Pixel>& leftPixels, const vector<Pixel>& rightPixels , vec3 color, vec3 normal)
{
	for(int i = 0; i < leftPixels.size(); i++)
	{
		DrawLineSDL(screen, leftPixels[i],rightPixels[i],color, normal);
	}
}

void DrawPolygon( const vector<Vertex>& vertices , vec3 color, vec3 normal)
{
	int V = vertices.size();
	vector<Pixel> vertexPixels( V );

	for( int i=0; i<V; ++i )
		VertexShader( vertices[i], vertexPixels[i] );

	vector<Pixel> leftPixels;
	vector<Pixel> rightPixels;

	ComputePolygonRows( vertexPixels, leftPixels, rightPixels );
	DrawRows( leftPixels, rightPixels , color, normal);
}
