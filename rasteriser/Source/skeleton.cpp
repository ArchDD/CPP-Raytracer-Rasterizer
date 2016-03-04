#include <iostream>
#include <glm/glm.hpp>
#include <SDL.h>
#include "SDLauxiliary.h"
#include "TestModel.h"

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::vec2;
using glm::ivec2;

/* ----------------------------------------------------------------------------*/
/* GLOBAL VARIABLES                                                            */

const int SCREEN_WIDTH = 500;
const int SCREEN_HEIGHT = 500;
SDL_Surface* screen;
int t;
float focalLength = 250.0f;

vec3 cameraPos( 0, 0, -2.0f );
mat3 cameraRot = mat3(0.0f);
float yaw = 0; // Yaw angle controlling camera rotation around y-axis

/* ----------------------------------------------------------------------------*/
/* FUNCTIONS                                                                   */
vector<Triangle> triangles;

void Update();
void Draw();
void VertexShader( const vec3& v, ivec2& p );
void Interpolate( ivec2 a, ivec2 b, vector<ivec2>& result );
void DrawLineSDL( SDL_Surface* surface, ivec2 a, ivec2 b, vec3 color );
void DrawPolygonEdges( const vector<vec3>& vertices );

int main( int argc, char* argv[] )
{
	screen = InitializeSDL( SCREEN_WIDTH, SCREEN_HEIGHT );
	t = SDL_GetTicks();	// Set start value for timer.

	// Generate the Cornell Box
	LoadTestModel( triangles );

	cameraRot[1][1] = 1.0f;

	while( NoQuitMessageSDL() )
	{
		Update();
		Draw();

		// Clear the screen before drawing the next frame
		for( int y=0; y<SCREEN_HEIGHT; ++y )
		{
			for( int x=0; x<SCREEN_WIDTH; ++x )
			{
				vec3 color( 0.0, 0.0, 0.0 );
				PutPixelSDL( screen, x, y, color );
			}
		}
	}

	SDL_SaveBMP( screen, "screenshot.bmp" );
	return 0;
}

void Update()
{
	// Compute frame time:
	int t2 = SDL_GetTicks();
	float dt = float(t2-t);
	t = t2;
	cout << "Render time: " << dt << " ms." << endl;

	// Adjust camera transform
	vec3 right(cameraRot[0][0], cameraRot[0][1], cameraRot[0][2]);
	vec3 down(cameraRot[1][0], cameraRot[1][1], cameraRot[1][2]);
	vec3 forward(cameraRot[2][0], cameraRot[2][1], cameraRot[2][2]);
	Uint8* keystate = SDL_GetKeyState( 0 );

	if( keystate[SDLK_UP] )
	{
		// Move camera forward
		cameraPos += 0.1f*forward;
	}
	else if( keystate[SDLK_DOWN] )
	{
		// Move camera backward
		cameraPos -= 0.1f*forward;
	}
	if( keystate[SDLK_LEFT] )
	{
		// Rotate camera to the left
		yaw += 0.1f;
	}
	else if( keystate[SDLK_RIGHT] )
	{
		// Rotate camera to the right
		yaw -= 0.1f;
	}

	// Update camera rotation matrix
	float c = cos(yaw);
	float s = sin(yaw);
	cameraRot[0][0] = c;
	cameraRot[0][2] = s;
	cameraRot[2][0] = -s;
	cameraRot[2][2] = c;
}

void Draw()
{
	if( SDL_MUSTLOCK(screen) )
		SDL_LockSurface(screen);

	for( size_t i = 0; i < triangles.size(); ++i )
	{
		// Get the 3 vertices of the triangle
		vector<vec3> vertices(3);
		vertices[0] = triangles[i].v0;
		vertices[1] = triangles[i].v1;
		vertices[2] = triangles[i].v2;

		DrawPolygonEdges( vertices );
	}

	if( SDL_MUSTLOCK(screen) )
		SDL_UnlockSurface(screen);

	SDL_UpdateRect( screen, 0, 0, 0, 0 );
}

// Compute 2D screen position from 3D position
void VertexShader( const vec3& v, ivec2& p )
{
	// We need to adjust the vertex positions based on the camera position and rotation
	vec3 v2(v);
	v2 = (v - cameraPos) * cameraRot;

	// Calculate 2D screen position and place (0,0) at top left of the screen
	p.x = (focalLength * (v2.x / v2.z)) + (SCREEN_WIDTH / 2.0f);
	p.y = (focalLength * (v2.y / v2.z)) + (SCREEN_HEIGHT / 2.0f);
}

// Draws a line between two points
void DrawLineSDL( SDL_Surface* surface, ivec2 a, ivec2 b, vec3 color )
{
	ivec2 delta = glm::abs( a - b );
	int pixels = glm::max( delta.x, delta.y ) + 1;

	vector<ivec2> line( pixels );
	Interpolate( a, b, line );

	for(int i = 0; i < pixels; ++i)
	{
		vec3 color(1,1,1);
		PutPixelSDL( surface, line[i].x, line[i].y, color );
	}
}

// Interpolates between two 2D vectors
void Interpolate( ivec2 a, ivec2 b, vector<ivec2>& result )
{
	int N = result.size();
	vec2 step = vec2(b-a) / float(max(N-1,1));
	vec2 current( a );
	for( int i=0; i<N; ++i )
	{
		result[i] = current;
		current += step;
	}
}

// Draw every polygon edge
void DrawPolygonEdges( const vector<vec3>& vertices )
{
	int V = vertices.size();
	// Transform each vertex from 3D world position to 2D image position:
	vector<ivec2> projectedVertices( V );
	for( int i=0; i<V; ++i )
	{
		VertexShader( vertices[i], projectedVertices[i] );
	}
	// Loop over all vertices and draw the edge from it to the next vertex:
	for( int i=0; i<V; ++i )
	{
		int j = (i+1)%V; // The next vertex
		vec3 color( 1, 1, 1 );
		DrawLineSDL( screen, projectedVertices[i], projectedVertices[j], color );
	}
}
