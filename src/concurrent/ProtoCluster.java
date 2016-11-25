package concurrent;

/**
 * Created by ctotolin on 24-Nov-16.
 */

import java.util.Arrays;

/**
 * global.Cluster class used temporarily during clustering.  Upon completion,
 * the array of ProtoClusters is transformed into an array of
 * Clusters.
 */
class ProtoCluster {

    // The previous iteration's cluster membership and
    // the current iteration's membership.  Compared to see if the
    // cluster has changed during the last iteration.
    private int[] mPreviousMembership;
    private int[] mCurrentMembership;
    private int mCurrentSize;

    // The cluster center.
    private double[] mCenter;

    // Born true, so the first call to updateDistances() will set all the
    // distances.
    private boolean mUpdateFlag = true;
    // Whether or not this cluster takes part in the operations.
    private boolean mConsiderForAssignment = true;

    /**
     * Constructor
     *
     * @param center  the initial cluster center.
     * @param coordIndex  the initial member.
     */
    ProtoCluster(double[] center, int coordIndex) {
        mCenter = (double[]) center.clone();
        // No previous membership.
        mPreviousMembership = new int[0];
        // Provide space for 10 members to be added initially.
        mCurrentMembership = new int[10];
        mCurrentSize = 0;
        add(coordIndex);
    }

    /**
     * Get the members of this protocluster.
     *
     * @return an array of coordinate indices.
     */
    int[] getMembership() {
        trimCurrentMembership();
        return mCurrentMembership;
    }

    /**
     * Get the protocluster's center.
     *
     * @return
     */
    double[] getCenter() {
        return mCenter;
    }

    /**
     * Reduces the length of the array of current members to
     * the number of members.
     */
    void trimCurrentMembership() {
        if (mCurrentMembership.length > mCurrentSize) {
            int[] temp = new int[mCurrentSize];
            System.arraycopy(mCurrentMembership, 0, temp, 0, mCurrentSize);
            mCurrentMembership = temp;
        }
    }

    /**
     * Add a coordinate to the protocluster. Note that this
     * method has to be synchronized, because multiple threads
     * may be adding members to the cluster.
     *
     * @param ndx index of the coordinate to be added.
     */
    synchronized void add(int ndx) {
        // Ensure there's space to add the new member.
        if (mCurrentSize == mCurrentMembership.length) {
            // If not, double the size of mCurrentMembership.
            int newCapacity = Math.max(10, 2*mCurrentMembership.length);
            int[] temp = new int[newCapacity];
            System.arraycopy(mCurrentMembership, 0, temp, 0, mCurrentSize);
            mCurrentMembership = temp;
        }
        // Add the index.
        mCurrentMembership[mCurrentSize++] = ndx;
    }

    /**
     * Does the protocluster contain any members?
     *
     * @return true if the cluster is empty.
     */
    boolean isEmpty() {
        return mCurrentSize == 0;
    }

    /**
     * Compares the previous and the current membership.
     * Sets the update flag to true if the membership
     * changed in the previous call to makeAssignments().
     */
    void setUpdateFlag() {
        // Trim the current membership array length down to the
        // number of members.
        trimCurrentMembership();
        // Since members may have been added by multiple threads, they
        // are probably not in order.  They must be sorted in order to
        // do a valid comparison with mPreviousMembership.
        Arrays.sort(mCurrentMembership);
        mUpdateFlag = false;
        if (mPreviousMembership.length == mCurrentSize) {
            for (int i=0; i<mCurrentSize; i++) {
                if (mPreviousMembership[i] != mCurrentMembership[i]) {
                    mUpdateFlag = true;
                    break;
                }
            }
        } else { // Number of members has changed.
            mUpdateFlag = true;
        }
    }

    /**
     * Clears the current membership after copying it to the
     * previous membership.
     */
    void checkPoint() {
        mPreviousMembership = mCurrentMembership;
        mCurrentMembership = new int[10];
        mCurrentSize = 0;
    }

    /**
     * Is this protocluster currently in contention?
     *
     * @return true if this cluster is still in the running.
     */
    boolean getConsiderForAssignment() {
        return mConsiderForAssignment;
    }

    /**
     * Set the flag to indicate that this protocluster is
     * in or out of contention.
     *
     * @param b
     */
    void setConsiderForAssignment(boolean b) {
        mConsiderForAssignment = b;
    }

    /**
     * Get the value of the update flag.  This value is
     * used to determine whether to update the cluster center and
     * whether to recompute distances to the cluster.
     *
     * @return the value of the update flag.
     */
    boolean needsUpdate() {
        return mUpdateFlag;
    }

    /**
     * Update the cluster center.
     *
     * @param coordinates the array of coordinates.
     */
    void updateCenter(double[][] coordinates) {
        Arrays.fill(mCenter, 0.0);
        if (mCurrentSize > 0) {
            for (int i=0; i<mCurrentSize; i++) {
                double[] coord = coordinates[mCurrentMembership[i]];
                for (int j=0; j<coord.length; j++) {
                    mCenter[j] += coord[j];
                }
            }
            for (int i=0; i<mCenter.length; i++) {
                mCenter[i] /= mCurrentSize;
            }
        }
    }
}