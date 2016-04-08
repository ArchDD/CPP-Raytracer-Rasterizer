#include<iostream>
#include<fstream>
#include<stdio.h>
#include<vector>
#include<string>
#include <sstream>
#include <glm/glm.hpp>
#include <algorithm>

using namespace std;

class LoadSTL
{
public:
	void LoadSTLFile(std::vector<Triangle>& triangles )
	{
		float scale = 0.05f;
		ifstream inputStream;
		string line;
		inputStream.open("Source/enemy1.stl");
		glm::vec3 green(  0.5f, 0.5f, 0.5f );

		triangles.clear();
		triangles.reserve( 1000 );

		while (getline(inputStream, line))
		{
			if(line.find("outer") != string::npos)
			{
				glm::vec3 v1,v2,v3;

				for(int i = 0; i < 3; i++)
				{
					string vertexLine; 
					getline(inputStream, vertexLine);
					vector<string> vertexValues = split(vertexLine, ' ');

					float x, y, z;
					x = (float)atof(vertexValues[0].c_str());
					y = (float)atof(vertexValues[1].c_str());
					z = (float)atof(vertexValues[2].c_str());

					if(i == 0)
						v1 = glm::vec3(x,y,z);
					else if(i == 1)
						v2 = glm::vec3(x,y,z);
					else 
						v3 = glm::vec3(x,y,z);
				}
				triangles.push_back( Triangle( v1, v2, v3, green ) );
			}
			
		}

		for( size_t i=0; i<triangles.size(); ++i )
		{

			triangles[i].v0.x *= -scale;
			triangles[i].v1.x *= -scale;
			triangles[i].v2.x *= -scale;

			triangles[i].v0.z *= -scale;
			triangles[i].v1.z *= -scale;
			triangles[i].v2.z *= -scale;

			triangles[i].v0.y *= -scale;
			triangles[i].v1.y *= -scale;
			triangles[i].v2.y *= -scale;

			triangles[i].ComputeNormal();
		}
	}

	// You could also take an existing vector as a parameter.
	vector<string> split(string str, char delimiter) 
	{
	  vector<string> internal;
	  stringstream ss(str); // Turn the string into a stream.
	  string tok;
	  
	  while(getline(ss, tok, delimiter)) 
	  {
	    if(tok.size() > 0 && tok != "vertex")
	    	internal.push_back(tok);
	  }
	  
	  return internal;
	}
};