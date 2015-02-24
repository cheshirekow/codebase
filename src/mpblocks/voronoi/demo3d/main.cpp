/*
 *  Copyright (C) 2012 Josh Bialkowski (jbialk@mit.edu)
 *
 *  This file is part of convex_hull.
 *
 *  convex_hull is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  convex_hull is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Fontconfigmm.  If not, see <http://www.gnu.org/licenses/>.
 */
/**
 *  \file   main.cpp
 *
 *  \date   Aug 19, 2012
 *  \author Josh Bialkowski (jbialk@mit.edu)
 *  \brief  
 */

#include <mpblocks/convex_hull/clarkson.hpp>
#include <OgreHardwareVertexBuffer.h>

#include "BaseApplication.h"

namespace ch = mpblocks::convex_hull::clarkson;

typedef   ch::Triangulation<double,3> Triangulation_t;
typedef   Eigen::Vector3d             Point_t;
typedef   std::list<Point_t*>         PtList_t;
typedef   std::set<Point_t*>          PtSet_t;
typedef   std::map<Point_t*,int>      PtMap_t;

class TutorialApplication : public BaseApplication
{

    private:
        bool             m_showHull;
        Ogre::SceneNode* m_pointsRoot;
        Ogre::SceneNode* m_hullRoot;
        Ogre::MaterialPtr  m_wireframe;
        Triangulation_t  m_ch;
        int              m_nPoints;

        PtList_t         m_ptList;
        PtSet_t          m_ptSet;
        PtMap_t          m_ptMap;

    public:
        TutorialApplication(void):m_showHull(true){}
        virtual ~TutorialApplication(void){}

    protected:
        virtual void createScene(void);
        virtual bool keyReleased( const OIS::KeyEvent& arg );
        void         generate();

};


//-------------------------------------------------------------------------------------
void TutorialApplication::createScene(void)
{
    m_nPoints = 10;

    m_wireframe =
        Ogre::MaterialManager::getSingleton()
            .create("BlackWireframe",
                Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
    m_wireframe->setDiffuse( Ogre::ColourValue(0,1,1) );
    m_wireframe->setCullingMode( Ogre::CULL_NONE );
    m_wireframe->getTechnique(0)->getPass(0)->setPolygonMode(Ogre::PM_WIREFRAME);

    // Set the scene's ambient light
    mSceneMgr->setAmbientLight(Ogre::ColourValue(0.5f, 0.5f, 0.5f));

    // Create an Entity
    //Ogre::Entity* ogreHead = mSceneMgr->createEntity("Head", "ogrehead.mesh");

    // Create a SceneNode and attach the Entity to it
    //Ogre::SceneNode* headNode = mSceneMgr->getRootSceneNode()->createChildSceneNode("HeadNode");
    //headNode->attachObject(ogreHead);

    // Create a Light and set its position
    Ogre::Light* light = mSceneMgr->createLight("MainLight");
    light->setPosition(20.0f, 80.0f, 50.0f);

    // create scene nodes for the point set and the hull
    m_pointsRoot = mSceneMgr->getRootSceneNode()->createChildSceneNode("points_root");
    m_hullRoot   = mSceneMgr->getRootSceneNode()->createChildSceneNode("hull_root");
}


bool TutorialApplication::keyReleased( const OIS::KeyEvent &arg )
{
    switch(arg.key)
    {
        case OIS::KC_C:
            m_showHull = !m_showHull;
            break;

        case OIS::KC_H:
            generate();
            break;

        case OIS::KC_P:
            m_nPoints++;
            generate();
            break;

        case OIS::KC_M:
            m_nPoints--;
            generate();
            break;

        default:
            return BaseApplication::keyReleased(arg);
            break;
    }

    m_hullRoot->setVisible(m_showHull);
    return true;
}

void TutorialApplication::generate()
{
    int     nPoints = m_nPoints;
    double  scale   = 0.05;
    double  grow    = 100;

    // clear the triangulation
    m_ch.reset();

    // delete meshes
    m_pointsRoot->removeAndDestroyAllChildren();
    m_hullRoot->removeAndDestroyAllChildren();

    // delete old points
    for( PtList_t::iterator ipPoint = m_ptList.begin();
            ipPoint != m_ptList.end(); ipPoint++)
    {
        delete *ipPoint;
    }

    m_ptList.clear();
    m_ptSet.clear();
    m_ptMap.clear();


    for(int i=0; i < nPoints; i++)
    {
        Point_t* ptr_x = new Point_t;
        m_ptList.push_back(ptr_x);

        Point_t& x = *ptr_x;
        for(int j=0; j < 3; j++)
            x[j] = grow * ( rand() / (double)RAND_MAX );

        m_ch.insert(ptr_x);
        Ogre::Entity* sphere = mSceneMgr->createEntity(
                                Ogre::SceneManager::PT_SPHERE);
        Ogre::SceneNode* node = m_pointsRoot->createChildSceneNode(
                                        Ogre::Vector3(x[0],x[1],x[2]));
        node->setScale(scale,scale,scale);
        node->attachObject(sphere);
    }

    // now we generate a mesh from the faces of the convex hull, first,
    // we iterate over the convex hull faces and add all their vertices to
    // the vertex set
    for( Triangulation_t::SimplexSet_t::iterator ipSimplex = m_ch.hull.begin();
            ipSimplex != m_ch.hull.end(); ipSimplex++ )
    {
        for(int i=1; i < 4; i++)
            m_ptSet.insert( (*ipSimplex)->vertices[i] );
    }

    // now we iterate over vertices of the hull and assign each of them an
    // index (this is used for the vertex buffer
    std::cerr << "Point map:\n";
    int i=0;
    for( PtSet_t::iterator ipPoint = m_ptSet.begin();
            ipPoint != m_ptSet.end(); ipPoint++ )
    {
        m_ptMap[*ipPoint] = i++;
        std::cerr << (void*)*ipPoint << " : " << m_ptMap[*ipPoint] << std::endl;
    }

    // lastly, we compute the geometric mean
    Point_t mean = Point_t::Zero();
    for( PtSet_t::iterator ipPoint = m_ptSet.begin();
            ipPoint != m_ptSet.end(); ipPoint++ )
    {
        mean += (**ipPoint) / m_ptSet.size();
    }

    // now we build the convex-hull mesh
    {
        /* create the mesh and a single sub mesh */
        Ogre::MeshPtr mesh =
                Ogre::MeshManager::getSingleton()
                .createOrRetrieve("HullMesh", "General",true).first;
        Ogre::SubMesh *subMesh = mesh->createSubMesh();

        /* create the vertex data structure */
        mesh->sharedVertexData = new Ogre::VertexData;
        mesh->sharedVertexData->vertexCount = m_ptSet.size();

        /* declare how the vertices will be represented */
        Ogre::VertexDeclaration *decl =
                mesh->sharedVertexData->vertexDeclaration;
        size_t offset = 0;

        /* the first three floats of each vertex represent the position */
        decl->addElement(0, offset, Ogre::VET_FLOAT3, Ogre::VES_POSITION);
        offset += Ogre::VertexElement::getTypeSize(Ogre::VET_FLOAT3);

        /* create the vertex buffer */
        Ogre::HardwareVertexBufferSharedPtr vertexBuffer =
                Ogre::HardwareBufferManager::getSingleton()
                .createVertexBuffer(offset,
                        mesh->sharedVertexData->vertexCount,
                        Ogre::HardwareBuffer::HBU_STATIC);

        /* lock the buffer so we can get exclusive access to its data */
        float *vertices = static_cast<float *>(
                    vertexBuffer->lock(Ogre::HardwareBuffer::HBL_NORMAL));

        /* populate the buffer with some data */
        int i=0;
        for( PtSet_t::iterator ipPoint = m_ptSet.begin();
                ipPoint != m_ptSet.end(); ipPoint++ )
        {
            Point_t& p = **ipPoint;
            for(int j=0; j < 3; j++)
                vertices[i++] = (float)p[j];
        }

        /* unlock the buffer */
        vertexBuffer->unlock();

        /* create the index buffer */
        Ogre::HardwareIndexBufferSharedPtr indexBuffer =
                Ogre::HardwareBufferManager::getSingleton()
                .createIndexBuffer(
                        Ogre::HardwareIndexBuffer::IT_16BIT,
                        m_ch.hull.size() * 3,               /// three indices per triangle
                        Ogre::HardwareBuffer::HBU_STATIC);

        /* lock the buffer so we can get exclusive access to its data */
        uint16_t *indices = static_cast<uint16_t *>(
                indexBuffer->lock(Ogre::HardwareBuffer::HBL_NORMAL));

        /* define our triangle */
        std::cerr << "There are " << m_ch.hull.size() << " faces on the convex hull" << std::endl;
        i = 0;
        for( Triangulation_t::SimplexSet_t::iterator ipSimplex = m_ch.hull.begin();
            ipSimplex != m_ch.hull.end(); ipSimplex++ )
        {
            std::cerr << "triangle " << i/3 << std::endl;
            Point_t* ps[3] = {0,0,0};

            for(int j=1; j < 4; j++)
                ps[j-1] = (*ipSimplex)->vertices[j];

            // first, we need to check that the order of the vertices emits
            // the correct normal
            Point_t v1 = *(ps[1]) - *(ps[0]);
            Point_t v2 = *(ps[2]) - *(ps[0]);
            Point_t vc = v1.cross(v2);
            Point_t vn = *(ps[0]) - mean;

            // now it should be that vc is in the same direction as vn (i.e.
            // the dot product is positive, so if it's not we swap the two
            if( vn.dot(vc) < 0 )
                std::swap( ps[1], ps[2] );

            for(int j=0; j < 3; j++)
            {
                Point_t* p = ps[j];
                std::cerr << "   " << (void*)p << " : "
                          << (uint16_t)m_ptMap[p] << "  "
                          << (*p).transpose() << std::endl;
                indices[i++] = (uint16_t)m_ptMap[p];
            }
        }

        /* unlock the buffer */
        indexBuffer->unlock();

        /* attach the buffers to the mesh */
        mesh->sharedVertexData->vertexBufferBinding->setBinding(0, vertexBuffer);
        subMesh->useSharedVertices = true;
        subMesh->indexData->indexBuffer = indexBuffer;
        subMesh->indexData->indexCount = m_ch.hull.size()*3;
        subMesh->indexData->indexStart = 0;

        /* set the bounds of the mesh */
        mesh->_setBounds(Ogre::AxisAlignedBox(0, 0, 0, grow,grow,grow));

        /* notify the mesh that we're all ready */
        mesh->load();

        /* you can now create an entity/scene node based on your mesh, e.g. */
        mSceneMgr->destroyEntity("HullEntity");
        Ogre::Entity *entity =
                mSceneMgr->createEntity("HullEntity", "HullMesh", "General");
        entity->setMaterial(m_wireframe);

        m_hullRoot->attachObject(entity);
    }
}


int main(int argc, char *argv[])
{
    // Create application object
    TutorialApplication app;

    try
    {
        app.go();
    } catch( Ogre::Exception& e ) {
        std::cerr << "An exception has occured: " << e.getFullDescription().c_str() << std::endl;
    }

    return 0;
}







