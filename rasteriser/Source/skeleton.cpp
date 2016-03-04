#include <iostream>
#include <glm/glm.hpp>
#include <SDL.h>
#include "SDLauxiliary.h"
#include "TestModel.h"

using namespace std;
using glm::vec3;
using glm::mat3;
using glm::ivec2;

/* ----------------------------------------------------------------------------*/
/* GLOBAL VARIABLES                                                            */

const int SCREEN_WIDTH = 500;
const int SCREEN_HEIGHT = 500;
SDL_Surface* screen;
int t;
float focalLength = 100.0f;

vec3 cameraPos( 0, 0, -3.001 );

/* ----------------------------------------------------------------------------*/
/* FUNCTIONS                                                                   */
vector<Triangle> triangles;

void Update();
void Draw();
void VertexShader( const vec3& v, ivec2& p );

int main( int argc, char* argv[] )
{
	screen = InitializeSDL( SCREEN_WIDTH, SCREEN_HEIGHT );
	t = SDL_GetTicks();	// Set start value for timer.

	// Generate the Cornell Box
	LoadTestModel( triangles );

	while( NoQuitMessageSDL() )
	{
		Update();
		Draw();
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
}

void Draw()
{
	if( SDL_MUSTLOCK(screen) )
		SDL_LockSurface(screen);

	for( int i=0; i< triangles.size(); ++i )
	{
		vector<vec3> vertices(3);
		vertices[0] = triangles[i].v0;
		vertices[1] = triangles[i].v1;
		vertices[2] = triangles[i].v2;
		for(int v=0; v<3; ++v)
		{
			ivec2 projPos;
			VertexShader( vertices[v], projPos );
			vec3 color(1,1,1);
			PutPixelSDL( screen, projPos.x, projPos.y, color );
		}
	}

	if( SDL_MUSTLOCK(screen) )
		SDL_UnlockSurface(screen);

	SDL_UpdateRect( screen, 0, 0, 0, 0 );
}

void VertexShader( const vec3& v, ivec2& p )
{
	p.x = (focalLength * (v.x / v.z)) + (SCREEN_WIDTH / 2.0f);
	p.y = (focalLength * (v.y / v.z)) + (SCREEN_HEIGHT / 2.0f);
}
