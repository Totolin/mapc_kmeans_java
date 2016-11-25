package concurrent;

import global.Cluster;
import global.InsufficientMemoryException;
import global.KMeans;
import global.KMeansListener;

import java.util.*;
import java.util.concurrent.*;

/**
 * The version of K-means clustering adapted for true concurrency
 * or simultaneous multithreading (SMT).  The subtasks of
 * computing distances and making assignments are delegate to
 * a subtask manager which oversees a thread pool.
 */
public class ConcurrentKMeans implements KMeans {

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
    // The number of threads used to perform the subtasks.
    private int mThreadCount;
    // Subtask manager that handles the thread pool to which
    // time-consuming tasks are delegated.
    private SubtaskManager mSubtaskManager;
    
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
     * @param threadCount the number of threads to be used for computing time-consuming steps.
     */
    public ConcurrentKMeans(double[][] coordinates, int k, int maxIterations, 
            long randomSeed, int threadCount) {
        mCoordinates = coordinates;
        // Can't have more clusters than coordinates.
        mK = Math.min(k, mCoordinates.length);
        mMaxIterations = maxIterations;
        mRandomSeed = randomSeed;
        mThreadCount = threadCount;
    }

    /**
     * Constructor that uses the return from 
     * <tt>Runtime.getRuntime().availableProcessors()</tt> as the number
     * of threads for time-consuming steps.
     * 
     * @param coordinates two-dimensional array containing the coordinates to be clustered.
     * @param k  the number of desired clusters.
     * @param maxIterations the maximum number of clustering iterations.
     * @param randomSeed seed used with the random number generator.
     */
    public ConcurrentKMeans(double[][] coordinates, int k, int maxIterations, 
            long randomSeed) {
        this (coordinates, k, maxIterations, randomSeed, 
                Runtime.getRuntime().availableProcessors());
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
     * Removes a global.KMeansListener
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

            // Instantiate the subtask manager.
            mSubtaskManager = new SubtaskManager(mThreadCount);

            // Post a message about the state of concurrent subprocessing.
            if (mThreadCount > 1) {
                postKMeansMessage("... concurrent processing mode with "
                            + mThreadCount + " subtask threads");
            } else {
                postKMeansMessage("... non-concurrent processing mode");
            }

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
        
        if (mDistanceCache == null) {
            int numCoords = mCoordinates.length;
            int numClusters = mProtoClusters.length;
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
        
        // Bulk of the work is delegated to the
        // SubtaskManager.
        mSubtaskManager.computeDistances();
    }
    
    /** 
     * Assign each coordinate to the nearest cluster.  Called once
     * per iteration.  Returns the number of coordinates that have
     * changed their cluster membership.
     */
    private int makeAssignments() {

        // Checkpoint the clusters, so we'll be able to tell
        // which one have changed after all the assignments have been
        // made.
        int numClusters = mProtoClusters.length;
        for (int c = 0; c < numClusters; c++) {
            if (mProtoClusters[c].getConsiderForAssignment()) {
                mProtoClusters[c].checkPoint();
            }
        }

        // Bulk of the work is delegated to the SubtaskManager.
        mSubtaskManager.makeAssignments();
        // Get the number of moves from the SubtaskManager.
        return mSubtaskManager.numberOfMoves();
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
        if (mSubtaskManager != null) {
            mSubtaskManager.shutdown();
            mSubtaskManager = null;
        }
    }



    /**
     * The class which manages the SMT-adapted subtasks.
     */
    private class SubtaskManager {

        // Codes used to identify what step is being done.
        static final int DOING_NOTHING = 0;
        static final int COMPUTING_DISTANCES = 1;
        static final int MAKING_ASSIGNMENTS = 2;

        // What the object is currently doing. Set to one of the
        // three codes above.
        private int mDoing = DOING_NOTHING;

        // True if the at least one of the Workers is doing something.
        private boolean mWorking;

        // The executor that runs the Workers.
        // When in multiple processor mode, this is a ThreadPoolExecutor
        // with a fixed number of threads. In single-processor mode, it's
        // a simple implementation that calls the single worker's run
        // method directly.
        private Executor mExecutor;

        // A Barrier to wait on multiple Workers to finish up the current task.
        // In single-processor mode, there is no need for a barrier, so it
        // is not set.
        private CyclicBarrier mBarrier;

        // The worker objects which implement Runnable.
        private Worker[] mWorkers;

        /**
         * Constructor
         *
         * @param numThreads the number of worker threads to be used for
         *   the subtasks.
         */
        SubtaskManager(int numThreads) {

            if (numThreads <= 0) {
                throw new IllegalArgumentException("number of threads <= 0: "
                        + numThreads);
            }

            int coordCount = mCoordinates.length;

            // There would be no point in having more workers than
            // coordinates, since some of the workers would have nothing
            // to do.
            if (numThreads > coordCount) {
                numThreads = coordCount;
            }

            // Create the workers.
            mWorkers = new Worker[numThreads];

            // To hold the number of coordinates for each worker.
            int[] coordsPerWorker = new int[numThreads];

            // Initialize with the base amounts.
            Arrays.fill(coordsPerWorker, coordCount/numThreads);

            // There may be some leftovers, since coordCount may not be
            // evenly divisible by numWorkers. Add a coordinate to each
            // until all are covered.
            int leftOvers = coordCount - numThreads * coordsPerWorker[0];
            for (int i = 0; i < leftOvers; i++) {
                coordsPerWorker[i]++;
            }

            int startCoord = 0;
            // Instantiate the workers.
            for (int i = 0; i < numThreads; i++) {
                // Each worker needs to know its starting coordinate and the number of
                // coordinates it handles.
                mWorkers[i] = new Worker(startCoord, coordsPerWorker[i]);
                startCoord += coordsPerWorker[i];
            }

            if (numThreads == 1) { // Single-processor mode.

                // Create a simple executor that directly calls the single
                // worker's run method.  Do not set the barrier.
                mExecutor = new Executor() {
                    public void execute(Runnable runnable) {
                        if (!Thread.interrupted()) {
                            runnable.run();
                        } else {
                            throw new RejectedExecutionException();
                        }
                    }
                };

            } else { // Multiple-processor mode.

                // Need the barrier to notify the controlling thread when the
                // Workers are done.
                mBarrier = new CyclicBarrier(numThreads, new Runnable() {
                    public void run() {
                        // Method called after all workers have called await() on the
                        // barrier.  The call to workersDone()
                        // unblocks the controlling thread.
                        workersDone();
                    }
                });

                // Set the executor to a fixed thread pool with
                // threads that do not time out.
                mExecutor = Executors.newFixedThreadPool(numThreads);
            }
        }

        /**
         * Make the cluster assignments.
         *
         * @return true if nothing went wrong.
         */
        boolean makeAssignments() {
            mDoing = MAKING_ASSIGNMENTS;
            return work();
        }

        /**
         * Compute the distances between the coordinates and those centers with
         * update flags.
         *
         * @return true if nothing went wrong.
         */
        boolean computeDistances() {
            mDoing = COMPUTING_DISTANCES;
            return work();
        }

        /**
         * Perform the current subtask, waiting until all the workers
         * finish their part of the current task before returning.
         *
         * @return true if the subtask succeeded.
         */
        private boolean work() {
            boolean ok = false;
            // Set the working flag to true.
            mWorking = true;
            try {
                if (mBarrier != null) {
                    // Resets the barrier so it can be reused if
                    // this is not the first call to this method.
                    mBarrier.reset();
                }
                // Now execute the run methods on the Workers.
                for (int i = 0; i < mWorkers.length; i++) {
                    mExecutor.execute(mWorkers[i]);
                }
                if (mBarrier != null) {
                    // Block until the workers are done.  The barrier
                    // triggers the unblocking.
                    waitOnWorkers();
                    // If the isBroken() method of the barrier returns false,
                    // no problems.
                    ok = !mBarrier.isBroken();
                } else {
                    // No barrier, so the run() method of a single worker
                    // was called directly and everything must have worked
                    // if we made it here.
                    ok = true;
                }
            } catch (RejectedExecutionException ree) {
                // Possibly thrown by the executor.
            } finally {
                mWorking = false;
            }
            return ok;
        }

        /**
         * Called from work() to put the controlling thread into
         * wait mode until the barrier calls workersDone().
         */
        private synchronized void waitOnWorkers() {
            // It is possible for the workers to have finished so quickly that
            // workersDone() has already been called.  Since workersDone() sets
            // mWorking to false, check this flag before going into wait mode.
            // Not doing so could result in hanging the SubtaskManager.
            while (mWorking) {
                try {
                    // Blocks until workersDone() is called.
                    wait();
                } catch (InterruptedException ie) {
                    // mBarrier.isBroken() will return true.
                    break;
                }
            }
        }

        /**
         * Notifies the controlling thread that it can come out of
         * wait mode.
         */
        private synchronized void workersDone() {
            // If this gets called before waitOnWorkers(), setting this
            // to false prevents waitOnWorkers() from entering a
            // permanent wait.
            mWorking = false;
            notifyAll();
        }

        /**
         * Shutdown the thread pool when k-means is finished.
         */
        void shutdown() {
            if (mExecutor instanceof ThreadPoolExecutor) {
                // This terminates the threads in the thread pool.
                ((ThreadPoolExecutor) mExecutor).shutdownNow();
            }
        }

        /**
         * Returns the number of cluster assignment changes made in the
         * previous call to makeAssignments().
         */
        int numberOfMoves() {
            // Sum the contributions from the workers.
            int moves = 0;
            for (int i=0; i<mWorkers.length; i++) {
                moves += mWorkers[i].numberOfMoves();
            }
            return moves;
        }

        /**
         * The class which does the hard work of the subtasks.
         */
        private class Worker implements Runnable {

            // Defines range of coordinates to cover.
            private int mStartCoord, mNumCoords;

            // Number of moves made by this worker in the last call
            // to workerMakeAssignments().  The SubtaskManager totals up
            // this value from all the workers in numberOfMoves().
            private int mMoves;

            /**
             * Constructor
             *
             * @param startCoord index of the first coordinate covered by
             *   this Worker.
             * @param numCoords the number of coordinates covered.
             */
            Worker(int startCoord, int numCoords) {
                mStartCoord = startCoord;
                mNumCoords = numCoords;
            }

            /**
             * Returns the number of moves this worker made in the last
             * execution of workerMakeAssignments()
             */
            int numberOfMoves() {
                return mMoves;
            }

            /**
             * The run method.  It accesses the SubtaskManager field mDoing
             * to determine what subtask to perform.
             */
            public void run() {

                try {
                    switch (mDoing) {
                    case COMPUTING_DISTANCES:
                        workerComputeDistances();
                        break;
                    case MAKING_ASSIGNMENTS:
                        workerMakeAssignments();
                        break;
                    }
                } finally {
                    // If there's a barrier, call its await() method.  To ensure it
                    // gets done, it's placed in the finally clause.
                    if (mBarrier != null) {
                        try {
                            mBarrier.await();
                        // barrier.isBroken() will return true if either of these
                        // exceptions happens, so the SubtaskManager will detect
                        // the problem.
                        } catch (InterruptedException | BrokenBarrierException ignored) {
                        }
                    }
                }
            }

            /**
             * Compute the distances for the covered coordinates
             * to the updated centers.
             */
            private void workerComputeDistances() {
                int lim = mStartCoord + mNumCoords;
                for (int i = mStartCoord; i < lim; i++) {
                    int numClusters = mProtoClusters.length;
                    for (int c = 0; c < numClusters; c++) {
                        ProtoCluster cluster = mProtoClusters[c];
                        if (cluster.getConsiderForAssignment() && cluster.needsUpdate()) {
                            mDistanceCache[i][c] = distance(mCoordinates[i], cluster.getCenter());
                        }
                    }
                }
            }

            /**
             * Assign each covered coordinate to the nearest cluster.
             */
            private void workerMakeAssignments() {
                mMoves = 0;
                int lim = mStartCoord + mNumCoords;
                for (int i = mStartCoord; i < lim; i++) {
                    int c = nearestCluster(i);
                    mProtoClusters[c].add(i);
                    if (mClusterAssignments[i] != c) {
                        mClusterAssignments[i] = c;
                        mMoves++;
                    }
                }
            }
        }
    }
}
