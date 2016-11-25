package basic;

import global.Cluster;
import global.InsufficientMemoryException;
import global.KMeans;
import global.KMeansListener;

import java.util.*;

/**
 * Basic implementation of K-means clustering.  Since it's a Runnable, it's 
 * designed to be executed by a dedicated thread, but that thread
 * does not create any other threads to divide up the work.
 */
public class BasicKMeans implements KMeans {

    // Temporary clusters used during the clustering process.  Converted to
    // an array of the simpler class global.Cluster at the conclusion.
    private ProtoCluster[] mProtoClusters;

    // Cache of coordinate-to-cluster distances. Number of entries = 
    // number of clusters X number of coordinates.
    private double[][] mDistanceCache;

    // Used in makeAssignments() to figure out how many moves are made
    // during each iteration -- the cluster assignment for coordinate n is
    // found in mClusterAssignments[n] where the N coordinates are numbered
    // 0 ... (N-1)
    private int[] mClusterAssignments;

    // 2D array holding the coordinates to be clustered.
    private double[][] mCoordinates;
    // The desired number of clusters and maximum number
    // of iterations.
    private int mK, mMaxIterations;
    // Seed for the random number generator used to select
    // coordinates for the initial cluster centers.
    private long mRandomSeed;
    
    // An array of global.Cluster objects: the output of k-means.
    private Cluster[] mClusters;

    // Listeners to be notified of significant happenings.
    private List<KMeansListener> mListeners = new ArrayList<KMeansListener>(1);
    
    /**
     * Constructor
     * 
     * @param coordinates two-dimensional array containing the coordinates to be clustered.
     * @param k  the number of desired clusters.
     * @param maxIterations the maximum number of clustering iterations.
     * @param randomSeed seed used with the random number generator.
     */
    public BasicKMeans(double[][] coordinates, int k, int maxIterations, 
            long randomSeed) {
        mCoordinates = coordinates;
        // Can't have more clusters than coordinates.
        mK = Math.min(k, mCoordinates.length);
        mMaxIterations = maxIterations;
        mRandomSeed = randomSeed;
    }

    /** 
     * Adds a global.KMeansListener to be notified of significant happenings.
     * 
     * @param l  the listener to be added.
     */
    public void addKMeansListener(KMeansListener l) {
        synchronized (mListeners) {
            if (!mListeners.contains(l)) {
                mListeners.add(l);
            }
        }
    }
    
    /**
     * Removes a global.KMeansListener from the listener list.
     * 
     * @param l the listener to be removed.
     */
    public void removeKMeansListener(KMeansListener l) {
        synchronized (mListeners) {
            mListeners.remove(l);
        }
    }
    
    /**
     * Posts a message to registered KMeansListeners.
     * 
     * @param message
     */
    private void postKMeansMessage(String message) {
        if (mListeners.size() > 0) {
            synchronized (mListeners) {
                int sz = mListeners.size();
                for (int i=0; i<sz; i++) {
                    mListeners.get(i).kmeansMessage(message);
                }
            }
        }
    }
    
    /**
     * Notifies registered listeners that k-means is complete.
     * 
     * @param clusters the output of clustering.
     * @param executionTime the number of milliseconds taken to cluster.
     */
    private void postKMeansComplete(Cluster[] clusters, long executionTime) {
        if (mListeners.size() > 0) {
            synchronized (mListeners) {
                int sz = mListeners.size();
                for (int i=0; i<sz; i++) {
                    mListeners.get(i).kmeansComplete(clusters, executionTime);
                }
            }
        }
    }
    
    /**
     * Notifies registered listeners that k-means has failed because of
     * a Throwable caught in the run method.
     * 
     * @param err 
     */
    private void postKMeansError(Throwable err) {
        if (mListeners.size() > 0) {
            synchronized (mListeners) {
                int sz = mListeners.size();
                for (int i=0; i<sz; i++) {
                    mListeners.get(i).kmeansError(err);
                }
            }
        }
    }

    /**
     * Get the clusters computed by the algorithm.  This method should
     * not be called until clustering has completed successfully.
     * 
     * @return an array of global.Cluster objects.
     */
    public Cluster[] getClusters() {
        return mClusters;
    }
    
    /**
     * Run the clustering algorithm.
     */
    public void run() {

        try {
            
            // Note the start time.
            long startTime = System.currentTimeMillis();
            
            postKMeansMessage("K-Means clustering started");
            
            // Randomly initialize the cluster centers creating the
            // array mProtoClusters.
            initCenters();
            
            postKMeansMessage("... centers initialized");

            // Perform the initial computation of distances.
            computeDistances();

            // Make the initial cluster assignments.
            makeAssignments();

            // Number of moves in the iteration and the iteration counter.
            int moves = 0, it = 0;
            
            // Main Loop:
            //
            // Two stopping criteria:
            // - no moves in makeAssignments 
            //   (moves == 0)
            // OR
            // - the maximum number of iterations has been reached
            //   (it == mMaxIterations)
            //
            do {

                // Compute the centers of the clusters that need updating.
                computeCenters();
                
                // Compute the stored distances between the updated clusters and the
                // coordinates.
                computeDistances();

                // Make this iteration's assignments.
                moves = makeAssignments();

                it++;
                
                postKMeansMessage("... iteration " + it + " moves = " + moves);

            } while (moves > 0 && it < mMaxIterations);

            // Transform the array of ProtoClusters to an array
            // of the simpler class global.Cluster.
            mClusters = generateFinalClusters();
            
            long executionTime = System.currentTimeMillis() - startTime;
            
            postKMeansComplete(mClusters, executionTime);
            
        } catch (Throwable t) {
           
            postKMeansError(t);
            
        } finally {

            // Clean up temporary data structures used during the algorithm.
            cleanup();

        }
    }

    /**
     * Randomly select coordinates to be the initial cluster centers.
     */
    private void initCenters() {

        Random random = new Random(mRandomSeed);
        
        int coordCount = mCoordinates.length;

        // The array mClusterAssignments is used only to keep track of the cluster 
        // membership for each coordinate.  The method makeAssignments() uses it
        // to keep track of the number of moves.
        if (mClusterAssignments == null) {
            mClusterAssignments = new int[coordCount];
            // Initialize to -1 to indicate that they haven't been assigned yet.
            Arrays.fill(mClusterAssignments, -1);
        }

        // Place the coordinate indices into an array and shuffle it.
        int[] indices = new int[coordCount];
        for (int i = 0; i < coordCount; i++) {
            indices[i] = i;
        }
        for (int i = 0, m = coordCount; m > 0; i++, m--) {
            int j = i + random.nextInt(m);
            if (i != j) {
                // Swap the indices.
                indices[i] ^= indices[j];
                indices[j] ^= indices[i];
                indices[i] ^= indices[j];
            }
        }

        mProtoClusters = new ProtoCluster[mK];
        for (int i=0; i<mK; i++) {
            int coordIndex = indices[i];
            mProtoClusters[i] = new ProtoCluster(mCoordinates[coordIndex], coordIndex);
            mClusterAssignments[indices[i]] = i;
        }
    }

    /**
     * Recompute the centers of the protoclusters with 
     * update flags set to true.
     */
    private void computeCenters() {
        
        int numClusters = mProtoClusters.length;
        
        // Sets the update flags of the protoclusters that haven't been deleted and
        // whose memberships have changed in the iteration just completed.
        //
        for (int c = 0; c < numClusters; c++) {
            ProtoCluster cluster = mProtoClusters[c];
            if (cluster.getConsiderForAssignment()) {
                if (!cluster.isEmpty()) {
                    // This sets the protocluster's update flag to
                    // true only if its membership changed in last call
                    // to makeAssignments().  
                    cluster.setUpdateFlag();
                    // If the update flag was set, update the center.
                    if (cluster.needsUpdate()) {
                        cluster.updateCenter(mCoordinates);
                    }
                } else {
                    // When a cluster loses all of its members, it
                    // falls out of contention.  So it is possible for
                    // k-means to return fewer than k clusters.
                    cluster.setConsiderForAssignment(false);
                }
            }
        }
    }

    /** 
     * Compute distances between coodinates and cluster centers,
     * storing them in the distance cache.  Only distances that
     * need to be computed are computed.  This is determined by
     * distance update flags in the protocluster objects.
     */
    private void computeDistances() throws InsufficientMemoryException {
        
        int numCoords = mCoordinates.length;
        int numClusters = mProtoClusters.length;

        if (mDistanceCache == null) {
            // Explicit garbage collection to reduce likelihood of insufficient
            // memory.
            System.gc();
            // Ensure there is enough memory available for the distances.  
            // Throw an exception if not.
            long memRequired = 8L * numCoords * numClusters;
            if (Runtime.getRuntime().freeMemory() < memRequired) {
                throw new InsufficientMemoryException();
            }
            // Instantiate an array to hold the distances between coordinates
            // and cluster centers
            mDistanceCache = new double[numCoords][numClusters];
        }

        // For each coordinate, find out closest cluster
        for (int coord=0; coord < numCoords; coord++) {
            // Update the distances between the coordinate and all
            // clusters currently in contention with update flags set.
            for (int clust = 0; clust < numClusters; clust++) {
                ProtoCluster cluster = mProtoClusters[clust];
                if (cluster.getConsiderForAssignment() && cluster.needsUpdate()) {
                    mDistanceCache[coord][clust] =
                            distance(mCoordinates[coord], cluster.getCenter());
                }
            }
        }
    }
    
    /** 
     * Assign each coordinate to the nearest cluster.  Called once
     * per iteration.  Returns the number of coordinates that have
     * changed their cluster membership.
     */
    private int makeAssignments() {

        int moves = 0;
        int coordCount = mCoordinates.length;

        // Checkpoint the clusters, so we'll be able to tell
        // which ones have changed after all the assignments have been
        // made.
        int numClusters = mProtoClusters.length;
        for (int c = 0; c < numClusters; c++) {
            if (mProtoClusters[c].getConsiderForAssignment()) {
                mProtoClusters[c].checkPoint();
            }
        }

        // Now do the assignments.
        for (int i = 0; i < coordCount; i++) {
            int c = nearestCluster(i);
            mProtoClusters[c].add(i);
            if (mClusterAssignments[i] != c) {
                mClusterAssignments[i] = c;
                moves++;
            }
        }

        return moves;
    }

    /**
     * Find the nearest cluster to the coordinate identified by
     * the specified index.
     */
    private int nearestCluster(int ndx) {
        int nearest = -1;
        double min = Double.MAX_VALUE;
        int numClusters = mProtoClusters.length;
        for (int c = 0; c < numClusters; c++) {
            if (mProtoClusters[c].getConsiderForAssignment()) {
                double d = mDistanceCache[ndx][c];
                if (d < min) {
                    min = d;
                    nearest = c;
                }
            }
        }
        return nearest;
    }
 
    /**
     * Compute the euclidean distance between the two arguments.
     */
    private static double distance(double[] coord, double[] center) {
        int len = coord.length;
        double sumSquared = 0.0;
        for (int i=0; i<len; i++) {
            double v = coord[i] - center[i];
            sumSquared += v*v;
        }
        return Math.sqrt(sumSquared);
    }

    /**
     * Generate an array of global.Cluster objects from mProtoClusters.
     * 
     * @return array of global.Cluster object references.
     */
    private Cluster[] generateFinalClusters() {
        
        int numClusters = mProtoClusters.length;
        
        // Convert the proto-clusters to the final Clusters.
        //
        // - accumulate in a list.
        List<Cluster> clusterList = new ArrayList<Cluster>(numClusters);
        for (int c = 0; c < numClusters; c++) {
            ProtoCluster pcluster = mProtoClusters[c];
            if (!pcluster.isEmpty()) {
                Cluster cluster = new Cluster(pcluster.getMembership(), pcluster.getCenter());
                clusterList.add(cluster);
            }
        }
    
        // - convert list to an array.
        Cluster[] clusters = new Cluster[clusterList.size()];
        clusterList.toArray(clusters);

        return clusters;
    }
    
    /**
     * Clean up items used by the clustering algorithm that are no longer needed.
     */
    private void cleanup() {
        mProtoClusters = null;
        mDistanceCache = null;
        mClusterAssignments = null;
    }



}
